# Build on Windows only after placing the MSVC-built stablevqa.exe in this folder.
from pathlib import Path

root = Path(SPECPATH)
datas = [(str(root / "stablevqa.exe"), ".")]
for model in (root.parent / "onnx_models").glob("*.onnx"):
    datas.append((str(model), "onnx_models"))

a = Analysis([str(root / "stablevqa_launcher.py")], pathex=[str(root)], datas=datas,
             binaries=[], hiddenimports=[], hookspath=[], runtime_hooks=[], excludes=[])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="StableVQATool", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="StableVQATool")
