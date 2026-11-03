use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;
use strsim::{jaro_winkler, normalized_levenshtein};
use walkdir::{DirEntry, WalkDir};

const DB_MAGIC: &[u8; 8] = b"3E_DB01\0";

#[derive(Clone, Copy)]
struct DirMeta {
    mtime_secs: u64,
    mtime_nanos: u32,
    remaining_depth: u8,
}

struct Database {
    dirs: BTreeMap<PathBuf, DirMeta>,
    binaries: BTreeSet<PathBuf>,
}

fn get_db_path() -> PathBuf {
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("3e").join("3e.db")
}

fn get_mtime(meta: &fs::Metadata) -> (u64, u32) {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| (d.as_secs(), d.subsec_nanos()))
        .unwrap_or((0, 0))
}

fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(windows)]
    {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "exe" | "bat" | "cmd" | "com"))
            .unwrap_or(false)
    }
    #[cfg(not(any(unix, windows)))]
    {
        true
    }
}

fn is_ignored_dir_name(name: &str) -> bool {
    matches!(
        name,
        ".git" | ".hg" | ".svn" | "node_modules" | "deps" | "incremental" | "proc" | "sys" | "dev"
    )
}

fn is_ignored_dir(entry: &DirEntry) -> bool {
    if !entry.file_type().is_dir() || entry.depth() == 0 {
        return false;
    }
    entry
        .file_name()
        .to_str()
        .map(is_ignored_dir_name)
        .unwrap_or(false)
}

fn search_roots() -> Vec<(PathBuf, u8)> {
    let mut roots: Vec<(PathBuf, u8)> = Vec::new();

    if let Ok(cwd) = env::current_dir() {
        roots.push((cwd, 16));
    }

    if let Some(home) = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")) {
        roots.push((PathBuf::from(home), 5));
    }

    if let Some(path_var) = env::var_os("PATH") {
        for dir in env::split_paths(&path_var) {
            roots.push((dir, 1));
        }
    }

    #[cfg(unix)]
    {
        for dir in ["/tmp", "/opt", "/usr/local/bin", "/usr/bin", "/bin"] {
            roots.push((PathBuf::from(dir), 3));
        }
    }

    roots
}

fn front_compress<W: Write>(writer: &mut W, items: &[String]) -> io::Result<()> {
    let count = items.len() as u32;
    writer.write_all(&count.to_le_bytes())?;

    let mut prev: &[u8] = b"";
    for item in items {
        let bytes = item.as_bytes();
        let prefix_len = prev
            .iter()
            .zip(bytes.iter())
            .take_while(|(a, b)| a == b)
            .count()
            .min(u16::MAX as usize);
        let suffix = &bytes[prefix_len..];
        let suffix_len = suffix.len().min(u16::MAX as usize);

        writer.write_all(&(prefix_len as u16).to_le_bytes())?;
        writer.write_all(&(suffix_len as u16).to_le_bytes())?;
        writer.write_all(&suffix[..suffix_len])?;
        prev = bytes;
    }

    Ok(())
}

fn front_decompress<R: Read>(reader: &mut R) -> io::Result<Vec<String>> {
    let mut count_buf = [0u8; 4];
    reader.read_exact(&mut count_buf)?;
    let count = u32::from_le_bytes(count_buf) as usize;

    let mut result = Vec::with_capacity(count);
    let mut current = Vec::new();

    for _ in 0..count {
        let mut header = [0u8; 4];
        reader.read_exact(&mut header)?;
        let prefix_len = u16::from_le_bytes([header[0], header[1]]) as usize;
        let suffix_len = u16::from_le_bytes([header[2], header[3]]) as usize;

        if prefix_len > current.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid prefix length"));
        }

        current.truncate(prefix_len);
        let old_len = current.len();
        current.resize(old_len + suffix_len, 0);
        reader.read_exact(&mut current[old_len..])?;

        let s = String::from_utf8(current.clone())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        result.push(s);
    }

    Ok(result)
}

impl Database {
    fn build_full() -> Self {
        let mut db = Self {
            dirs: BTreeMap::new(),
            binaries: BTreeSet::new(),
        };

        for (root, max_depth) in search_roots() {
            db.scan_tree(&root, max_depth);
        }

        db
    }

    fn scan_tree(&mut self, root: &Path, max_depth: u8) {
        let mut stack = vec![(root.to_path_buf(), max_depth)];

        while let Some((dir, depth)) = stack.pop() {
            let Ok(meta) = fs::metadata(&dir) else {
                continue;
            };
            if !meta.is_dir() {
                continue;
            }

            let (mtime_secs, mtime_nanos) = get_mtime(&meta);
            if let Some(existing) = self.dirs.get(&dir) {
                if existing.remaining_depth >= depth
                    && existing.mtime_secs == mtime_secs
                    && existing.mtime_nanos == mtime_nanos
                {
                    continue;
                }
            }

            self.dirs.insert(
                dir.clone(),
                DirMeta {
                    mtime_secs,
                    mtime_nanos,
                    remaining_depth: depth,
                },
            );

            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };

            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(sym_meta) = fs::symlink_metadata(&path) else {
                    continue;
                };

                if sym_meta.is_dir() {
                    if depth > 1 {
                        let skip = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .map(is_ignored_dir_name)
                            .unwrap_or(false);
                        if !skip {
                            stack.push((path, depth - 1));
                        }
                    }
                } else if is_executable(&path) {
                    self.binaries.insert(path);
                }
            }
        }
    }

    fn refresh_smart(&mut self, db_path: &Path) {
        let mut dirty = false;

        if let Ok(cwd) = env::current_dir() {
            if !self.dirs.contains_key(&cwd) {
                self.scan_tree(&cwd, 16);
                dirty = true;
            }
        }

        let snapshot: Vec<(PathBuf, DirMeta)> =
            self.dirs.iter().map(|(k, v)| (k.clone(), *v)).collect();

        let mut removed_dirs = Vec::new();
        let mut changed_dirs = Vec::new();

        for (dir_path, saved_meta) in snapshot {
            match fs::metadata(&dir_path) {
                Ok(meta) if meta.is_dir() => {
                    let (secs, nanos) = get_mtime(&meta);
                    if secs != saved_meta.mtime_secs || nanos != saved_meta.mtime_nanos {
                        changed_dirs.push((dir_path, secs, nanos, saved_meta.remaining_depth));
                    }
                }
                _ => {
                    removed_dirs.push(dir_path);
                }
            }
        }

        if !removed_dirs.is_empty() {
            dirty = true;
            for dir in &removed_dirs {
                self.dirs.remove(dir);
            }
            let removed_set: BTreeSet<PathBuf> = removed_dirs.into_iter().collect();
            self.binaries.retain(|bin| {
                bin.parent()
                    .map(|p| !removed_set.contains(p))
                    .unwrap_or(true)
            });
        }

        if !changed_dirs.is_empty() {
            dirty = true;
            let changed_set: BTreeSet<PathBuf> =
                changed_dirs.iter().map(|(d, _, _, _)| d.clone()).collect();

            self.binaries.retain(|bin| {
                bin.parent()
                    .map(|p| !changed_set.contains(p))
                    .unwrap_or(true)
            });

            let mut new_subtrees = Vec::new();

            for (dir_path, secs, nanos, depth) in changed_dirs {
                self.dirs.insert(
                    dir_path.clone(),
                    DirMeta {
                        mtime_secs: secs,
                        mtime_nanos: nanos,
                        remaining_depth: depth,
                    },
                );

                if let Ok(entries) = fs::read_dir(&dir_path) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let Ok(sym_meta) = fs::symlink_metadata(&path) else {
                            continue;
                        };

                        if sym_meta.is_dir() {
                            if depth > 1 && !self.dirs.contains_key(&path) {
                                let skip = path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .map(is_ignored_dir_name)
                                    .unwrap_or(false);
                                if !skip {
                                    new_subtrees.push((path, depth - 1));
                                }
                            }
                        } else if is_executable(&path) {
                            self.binaries.insert(path);
                        }
                    }
                }
            }

            for (sub_root, sub_depth) in new_subtrees {
                self.scan_tree(&sub_root, sub_depth);
            }
        }

        if dirty {
            let _ = self.save(db_path);
        }
    }

    fn save(&self, db_path: &Path) -> io::Result<()> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = File::create(db_path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(DB_MAGIC)?;

        let mut dir_strings = Vec::with_capacity(self.dirs.len());
        let mut dir_metas = Vec::with_capacity(self.dirs.len());

        for (path, meta) in &self.dirs {
            if let Some(s) = path.to_str() {
                dir_strings.push(s.to_string());
                dir_metas.push(*meta);
            }
        }

        front_compress(&mut writer, &dir_strings)?;

        for meta in dir_metas {
            writer.write_all(&meta.mtime_secs.to_le_bytes())?;
            writer.write_all(&meta.mtime_nanos.to_le_bytes())?;
            writer.write_all(&[meta.remaining_depth])?;
        }

        let bin_strings: Vec<String> = self
            .binaries
            .iter()
            .filter_map(|p| p.to_str().map(|s| s.to_string()))
            .collect();

        front_compress(&mut writer, &bin_strings)?;
        writer.flush()
    }

    fn load(db_path: &Path) -> io::Result<Self> {
        let file = File::open(db_path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != DB_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid DB header"));
        }

        let dir_strings = front_decompress(&mut reader)?;
        let mut dirs = BTreeMap::new();

        for dir_str in dir_strings {
            let mut meta_buf = [0u8; 13];
            reader.read_exact(&mut meta_buf)?;
            let mtime_secs = u64::from_le_bytes(meta_buf[0..8].try_into().unwrap());
            let mtime_nanos = u32::from_le_bytes(meta_buf[8..12].try_into().unwrap());
            let remaining_depth = meta_buf[12];

            dirs.insert(
                PathBuf::from(dir_str),
                DirMeta {
                    mtime_secs,
                    mtime_nanos,
                    remaining_depth,
                },
            );
        }

        let bin_strings = front_decompress(&mut reader)?;
        let binaries: BTreeSet<PathBuf> = bin_strings.into_iter().map(PathBuf::from).collect();

        Ok(Self { dirs, binaries })
    }

    fn to_executable_map(&self) -> HashMap<String, PathBuf> {
        let mut map = HashMap::new();
        let cwd = env::current_dir().ok();

        for path in &self.binaries {
            let prioritize = cwd.as_ref().map(|c| path.starts_with(c)).unwrap_or(false);

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if prioritize || !map.contains_key(name) {
                    map.insert(name.to_string(), path.clone());
                }
            }
            #[cfg(windows)]
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if prioritize || !map.contains_key(stem) {
                    map.insert(stem.to_string(), path.clone());
                }
            }
        }

        map
    }
}

fn collect_executables_walkdir() -> HashMap<String, PathBuf> {
    let mut map = HashMap::new();

    for (root, max_depth) in search_roots() {
        let walker = WalkDir::new(root)
            .max_depth(max_depth as usize)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !is_ignored_dir(e));

        for entry in walker.flatten() {
            let path = entry.path();
            if is_executable(path) {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    map.entry(name.to_string()).or_insert_with(|| path.to_path_buf());
                }
                #[cfg(windows)]
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    map.entry(stem.to_string()).or_insert_with(|| path.to_path_buf());
                }
            }
        }
    }

    map
}

fn find_similar<'a>(
    query: &str,
    executables: &'a HashMap<String, PathBuf>,
) -> Option<(&'a String, &'a PathBuf)> {
    let query_lower = query.to_lowercase();
    let mut best_match: Option<(&String, &PathBuf)> = None;
    let mut best_score = 0.0_f64;

    for (name, path) in executables {
        let name_lower = name.to_lowercase();
        let jw = jaro_winkler(&query_lower, &name_lower);
        let nl = normalized_levenshtein(&query_lower, &name_lower);
        let mut score = jw.max(nl);

        if name_lower.contains(&query_lower) || query_lower.contains(&name_lower) {
            score = score.max(0.75);
        }

        if score > best_score {
            best_score = score;
            best_match = Some((name, path));
        }
    }

    if best_score >= 0.6 {
        best_match
    } else {
        None
    }
}

fn run_binary(path: &Path, args: &[&str]) {
    match Command::new(path).args(args).status() {
        Ok(_) => {}
        Err(e) => eprintln!("Failed to execute {}: {}", path.display(), e),
    }
}

fn main() {
    let db_path = get_db_path();
    let mut active_db: Option<Database> = None;

    if env::args().skip(1).any(|arg| {
        matches!(
            arg.as_str(),
            "3e.db" | "./3e.db" | "eee.db" | "./eee.db"
        )
    }) {
        let db = Database::build_full();
        let _ = db.save(&db_path);
        active_db = Some(db);
    } else if db_path.exists() {
        if let Ok(mut db) = Database::load(&db_path) {
            db.refresh_smart(&db_path);
            active_db = Some(db);
        }
    }

    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("3e> ");
        if io::stdout().flush().is_err() {
            break;
        }

        input.clear();
        let Ok(bytes_read) = stdin.read_line(&mut input) else {
            break;
        };
        if bytes_read == 0 {
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if matches!(trimmed, "exit" | "quit") {
            break;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let raw_cmd = parts[0];
        let args = &parts[1..];

        let clean_cmd = raw_cmd
            .strip_prefix("./")
            .or_else(|| raw_cmd.strip_prefix(".\\"))
            .unwrap_or(raw_cmd);

        if matches!(clean_cmd, "3e.db" | "eee.db") {
            let db = match active_db.take() {
                Some(mut existing) => {
                    existing.refresh_smart(&db_path);
                    let _ = existing.save(&db_path);
                    existing
                }
                None => {
                    let built = Database::build_full();
                    let _ = built.save(&db_path);
                    built
                }
            };
            active_db = Some(db);
            continue;
        }

        let direct_path = Path::new(raw_cmd);
        if direct_path.exists() && is_executable(direct_path) {
            run_binary(direct_path, args);
            continue;
        }

        let clean_path = Path::new(clean_cmd);
        if clean_path.exists() && is_executable(clean_path) {
            run_binary(clean_path, args);
            continue;
        }

        let target_name = clean_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(clean_cmd);

        if active_db.is_none() && db_path.exists() {
            if let Ok(db) = Database::load(&db_path) {
                active_db = Some(db);
            }
        }

        let executables = if let Some(ref mut db) = active_db {
            db.refresh_smart(&db_path);
            db.to_executable_map()
        } else {
            collect_executables_walkdir()
        };

        if let Some(path) = executables.get(target_name) {
            if is_executable(path) {
                run_binary(path, args);
                continue;
            }
        }

        if let Some((similar_name, similar_path)) = find_similar(target_name, &executables) {
            print!(
                "The executable file \"{}\" was not found.\nWould you like to run an executable file with a similar name – \"{}\" (Y/n): ",
                target_name, similar_name
            );
            let _ = io::stdout().flush();

            let mut confirm = String::new();
            if stdin.read_line(&mut confirm).is_ok() {
                let answer = confirm.trim().to_lowercase();
                if answer.is_empty() || answer == "y" || answer == "yes" {
                    run_binary(similar_path, args);
                }
            }
        } else {
            println!("The executable file \"{}\" was not found.", target_name);
        }
    }
}