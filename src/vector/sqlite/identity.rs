use super::super::in_memory::new_history_digest;
use super::super::{VectorIndexError, VectorResult};
use sha2::{Digest, Sha256};
use std::fs::Metadata;
use std::path::Path;

const STORAGE_IDENTITY_DOMAIN: &str = "a3s.memory.sqlite-vector-storage-identity.v1";

pub(super) fn storage_identity(path: &Path) -> VectorResult<String> {
    let metadata = std::fs::metadata(path).map_err(|_| {
        VectorIndexError::StorageFailed("could not inspect the SQLite database file".to_string())
    })?;
    platform_identity(path, &metadata).map_or_else(
        || Ok(new_history_digest()),
        |identity| {
            let mut hasher = Sha256::new();
            hasher.update(STORAGE_IDENTITY_DOMAIN.as_bytes());
            hasher.update([0]);
            hasher.update(identity);
            Ok(format!("sha256:{:x}", hasher.finalize()))
        },
    )
}

#[cfg(unix)]
fn platform_identity(_path: &Path, metadata: &Metadata) -> Option<Vec<u8>> {
    use std::os::unix::fs::MetadataExt;

    let mut identity = b"unix".to_vec();
    identity.extend(metadata.dev().to_le_bytes());
    identity.extend(metadata.ino().to_le_bytes());
    Some(identity)
}

#[cfg(windows)]
fn platform_identity(path: &Path, _metadata: &Metadata) -> Option<Vec<u8>> {
    use std::fs::File;
    use std::mem::MaybeUninit;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };

    let file = File::open(path).ok()?;
    let mut information = MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
    // SAFETY: `file` keeps the handle valid for the call and `information`
    // points to writable storage of the exact Win32 output type.
    let succeeded =
        unsafe { GetFileInformationByHandle(file.as_raw_handle(), information.as_mut_ptr()) };
    if succeeded == 0 {
        return None;
    }
    // SAFETY: Win32 reports success only after initializing the output value.
    let information = unsafe { information.assume_init() };
    let mut identity = b"windows".to_vec();
    identity.extend(information.dwVolumeSerialNumber.to_le_bytes());
    identity.extend(information.nFileIndexHigh.to_le_bytes());
    identity.extend(information.nFileIndexLow.to_le_bytes());
    identity.extend(information.ftCreationTime.dwHighDateTime.to_le_bytes());
    identity.extend(information.ftCreationTime.dwLowDateTime.to_le_bytes());
    Some(identity)
}

#[cfg(not(any(unix, windows)))]
fn platform_identity(_path: &Path, _metadata: &Metadata) -> Option<Vec<u8>> {
    None
}
