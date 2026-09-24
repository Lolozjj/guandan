"""从 Unity Addressables bundle 里提取资源（主要是 Texture2D 图片）。

掼蛋小程序是 Unity WebGL 构建，资源打包成 .bundle 存在本地缓存里。
这个脚本把 bundle 里的贴图解码成 PNG。

用法:
    python extract_unity_assets.py <bundle文件或目录> <输出目录> [--list-only]

例:
    python extract_unity_assets.py "<bundle路径>" assets/cards --list-only
    python extract_unity_assets.py "<bundle路径>" assets/cards
"""
import argparse
import re
from pathlib import Path

import UnityPy


def safe_name(s: str) -> str:
    """去掉文件名里的非法字符。"""
    return re.sub(r'[<>:"/\\|?*]', "_", str(s)).strip() or "unnamed"


def walk(bundle: Path, out_dir: Path, list_only: bool) -> None:
    env = UnityPy.load(str(bundle))
    counts: dict[str, int] = {}
    saved = 0

    for obj in env.objects:
        tname = obj.type.name
        counts[tname] = counts.get(tname, 0) + 1

        if list_only:
            continue

        try:
            if tname == "Texture2D":
                data = obj.read()
                img = data.image
                if img is None:
                    continue
                name = safe_name(getattr(data, "m_Name", f"tex_{obj.path_id}"))
                dest = out_dir / f"{name}.png"
                if dest.exists():
                    dest = out_dir / f"{name}_{obj.path_id}.png"
                img.save(dest)
                saved += 1

            elif tname == "TextAsset":
                data = obj.read()
                name = safe_name(getattr(data, "m_Name", f"text_{obj.path_id}"))
                raw = getattr(data, "m_Script", None)
                if raw is None:
                    continue
                if isinstance(raw, str):
                    raw = raw.encode("utf-8", errors="replace")
                (out_dir / f"{name}.txt").write_bytes(raw)
                saved += 1

            elif tname == "Sprite":
                # 精灵只是图集里的一个区域，单独存方便看边界
                data = obj.read()
                name = safe_name(getattr(data, "m_Name", f"sprite_{obj.path_id}"))
                img = getattr(data, "image", None)
                if img is None:
                    continue
                img.save(out_dir / f"_sprite_{name}.png")
                saved += 1

        except Exception as e:  # noqa: BLE001 - 单个对象失败不该中断整体提取
            print(f"    [skip] {tname} {getattr(obj, 'path_id', '?')}: {type(e).__name__}: {e}")

    print(f"  {bundle.name}")
    for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {t:<20} {n}")
    if not list_only:
        print(f"    -> 已保存 {saved} 个文件")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="bundle 文件或包含 bundle 的目录")
    ap.add_argument("out", help="输出目录")
    ap.add_argument("--list-only", action="store_true", help="只列出内容，不解码")
    ap.add_argument("--filter", default="", help="只处理文件名含该子串的 bundle")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    if not args.list_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    if src.is_file():
        bundles = [src]
    else:
        bundles = sorted(p for p in src.rglob("*.bundle"))
        if args.filter:
            bundles = [b for b in bundles if args.filter.lower() in b.name.lower()]
        if not bundles:
            raise SystemExit(f"[FAIL] {src} 下没找到 .bundle 文件")

    print(f"待处理 bundle: {len(bundles)} 个\n")
    for b in bundles:
        try:
            walk(b, out_dir, args.list_only)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {b.name}: {type(e).__name__}: {e}")

    if not args.list_only:
        print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
