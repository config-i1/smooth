# Windows DLL Handling Notes

## CI Tests vs Wheel Distribution

### For CI tests (current workflow)

The PATH fix is needed because we're running tests in an editable install where DLLs are loaded from the system. We add the vcpkg bin directory to PATH before running pytest:

```powershell
$env:PATH = "C:\vcpkg\installed\x64-windows-release\bin;$env:PATH"
```

### For cibuildwheel wheel building

This shouldn't be an issue if configured properly because platform-specific repair tools automatically bundle shared libraries into the wheel:

- **Linux**: `auditwheel` automatically bundles `.so` files into the wheel
- **macOS**: `delocate` automatically bundles `.dylib` files into the wheel
- **Windows**: `delvewheel` automatically bundles `.dll` files into the wheel

With cibuildwheel, you'd configure it to use `delvewheel` on Windows, which copies the OpenBLAS/LAPACK DLLs directly into the wheel. End users then get a self-contained wheel that doesn't require vcpkg.

## Recommended cibuildwheel Configuration

Add this to `pyproject.toml`:

```toml
[tool.cibuildwheel.windows]
before-build = "pip install delvewheel"
repair-wheel-command = "delvewheel repair -w {dest_dir} {wheel}"
```

You also need to ensure vcpkg's bin directory is in PATH during the repair step so `delvewheel` can find the DLLs to bundle:

```toml
[tool.cibuildwheel.windows]
environment = { PATH = "C:\\vcpkg\\installed\\x64-windows-release\\bin;$PATH" }
before-build = "pip install delvewheel"
repair-wheel-command = "delvewheel repair -w {dest_dir} {wheel}"
```

## Summary

- **CI test workflow**: Requires PATH fix for vcpkg DLLs
- **cibuildwheel release workflow**: Use `delvewheel` to bundle DLLs into the wheel automatically
