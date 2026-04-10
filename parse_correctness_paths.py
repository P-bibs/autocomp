#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


ITERATION_RE = re.compile(r"Iteration\s+(\d+)\s+of optimization:")
CODE_PATH_RE = re.compile(
    r"(/home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/"
    r"(kb_eval_[^/\s]+)/code_(\d+)\.py)"
)
BASE_DIR_RE = re.compile(
    r"/home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/"
    r"(kb_eval_[^/\s]+)"
)
PASSED_RE = re.compile(r"Kernel passed correctness for code (\d+)")
FAILED_RE = re.compile(r"Kernel did not pass correctness for code (\d+)")

TMP_FILES_ROOT = Path(
    "/home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files"
)
OUTPUT_DIR_INFO_RE = re.compile(r"(?:^|/)cuda_kb-(level\d+)_(\d+)(?:_|$)")


def iter_sections(text: str):
    matches = list(ITERATION_RE.finditer(text))
    if not matches:
        yield "preamble", text
        return

    if matches[0].start() > 0:
        yield "preamble", text[: matches[0].start()]

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        yield f"iteration {match.group(1)}", text[start:end]


def parse_section(section_text: str):
    code_to_path = {}
    bases_in_order = []

    for match in CODE_PATH_RE.finditer(section_text):
        full_path, base, code_num = match.groups()
        code_to_path[int(code_num)] = full_path
        bases_in_order.append(base)

    if not bases_in_order:
        bases_in_order.extend(BASE_DIR_RE.findall(section_text))

    section_base = bases_in_order[-1] if bases_in_order else None

    passed = []
    failed = []

    for regex, bucket in ((PASSED_RE, passed), (FAILED_RE, failed)):
        for match in regex.finditer(section_text):
            code_num = int(match.group(1))
            path = code_to_path.get(code_num)
            if path is None and section_base is not None:
                path = str(TMP_FILES_ROOT / section_base / f"code_{code_num}.py")
            if path is not None:
                bucket.append(path)

    return passed, failed


def parse_log(path: Path):
    text = path.read_text()
    passed = []
    failed = []

    for _, section_text in iter_sections(text):
        section_passed, section_failed = parse_section(section_text)
        passed.extend(section_passed)
        failed.extend(section_failed)

    return passed, failed


def dedupe_preserve_order(items):
    return list(dict.fromkeys(items))


def print_group(title: str, items):
    print(f"{title} ({len(items)}):")
    for item in items:
        print(item)
    if not items:
        print("(none)")


def parse_output_dir_info(output_dir: Path):
    match = OUTPUT_DIR_INFO_RE.search(str(output_dir))
    if match is None:
        raise ValueError(
            f"Could not extract level/problem id from output directory name: {output_dir}"
        )
    level, problem_id = match.groups()
    return level, problem_id


def find_log_file(output_dir: Path) -> Path:
    matches = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name.startswith("auto-comp")
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one auto-comp log in {output_dir}, found {len(matches)}"
        )
    return matches[0]


def copy_group(paths, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for index, source_name in enumerate(paths):
        source = Path(source_name)
        target = destination / f"{index}.py"
        source_text = source.read_text()
        target.write_text(f"# Original path: {source}\n{source_text}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse auto-comp output directories, print kernel source paths grouped "
            "by correctness result, and copy them into output_collated."
        )
    )
    parser.add_argument(
        "output_dirs",
        nargs="+",
        help="One or more output directories containing exactly one auto-comp log.",
    )
    parser.add_argument(
        "--collated-root",
        default="output_collated",
        help="Destination root for copied files. Defaults to output_collated.",
    )
    args = parser.parse_args()
    collated_root = Path(args.collated_root)
    (collated_root / "level1").mkdir(parents=True, exist_ok=True)
    (collated_root / "level2").mkdir(parents=True, exist_ok=True)

    for index, dir_name in enumerate(args.output_dirs):
        output_dir = Path(dir_name)
        log_path = find_log_file(output_dir)
        level, problem_id = parse_output_dir_info(output_dir)
        passed, failed = parse_log(log_path)
        passed = dedupe_preserve_order(passed)
        failed = dedupe_preserve_order(failed)
        correct_dir = collated_root / level / problem_id / "correct"
        incorrect_dir = collated_root / level / problem_id / "incorrect"
        copy_group(passed, correct_dir)
        copy_group(failed, incorrect_dir)

        if index:
            print()
        print(f"== {output_dir} ==")
        print(f"Log: {log_path}")
        print(f"Copied correct files to: {correct_dir}")
        print(f"Copied incorrect files to: {incorrect_dir}")
        print_group("Passed", passed)
        print()
        print_group("Failed", failed)


if __name__ == "__main__":
    main()
