"""List every class and function a header in src/ declares, with the comment above it.

Clang parses the headers, so this reads real declarations rather than guessing at them
with a regular expression - templates, defaulted members, deleted operators and
multi-line signatures all arrive resolved, and access and deletedness are answered by the
compiler rather than inferred.

It uses libclang's cursor traversal, which is lazy. The JSON route this replaced was
tried first and abandoned on a measurement: `clang -ast-dump=json` on a header that
includes torch/torch.h emits 6.35 GB in 68 seconds, which no Python process can load.
The same header through a cursor takes 9.4 seconds at constant memory, and that is what
lets the LibTorch and raylib headers be covered at all. Do not go back to the JSON dump.

`-fparse-all-comments` is what makes clang attach the plain `//` blocks this repository
writes. Doxygen ignores every one of them, which is why it is not used here.

Coverage is reported per file, and a declaration with no comment is a finding. Presence
is the whole of what it measures. Whether a comment says anything worth reading is
settled by reading it.

Usage:

    python docs/api_surface.py                     # writes docs/src_api.html
    python docs/api_surface.py --check             # report only, write nothing
    python docs/api_surface.py --only mcts.h       # one header, printed to the terminal

Exit codes: 0 every required declaration carries a comment, 1 at least one does not,
2 libclang is missing or a header could not be parsed at all.
"""

import argparse
import html
import sys
from pathlib import Path

import doc_page

try:
    import clang.cindex as cindex
except ImportError:  # reported by main rather than raised at import
    cindex = None

# Where the project's headers and its two external dependencies live. LibTorch is wired
# by absolute path in CMakeLists.txt and raylib comes from vcpkg; both are repeated here
# because clang is being run directly rather than through the build.
INCLUDE_PATHS = (
    "src",
    "D:/libtorch-cuda/libtorch/include",
    "D:/libtorch-cuda/libtorch/include/torch/csrc/api/include",
    "E:/dev/vcpkg/installed/x64-windows/include",
)

# What appears on the page. A typedef or a friend declaration is not something a caller
# documents, and listing it would dilute the coverage number.
LISTED_KINDS = frozenset(
    {
        "STRUCT_DECL",
        "CLASS_DECL",
        "ENUM_DECL",
        "FUNCTION_DECL",
        "CXX_METHOD",
        "CONSTRUCTOR",
        "FIELD_DECL",
        "VAR_DECL",
    }
)

# What a missing comment is a finding on. Fields and constants are listed but never
# required: this repository deliberately comments a group of members once, and the
# comment attaches to the first of them, so requiring one per field would report two
# false findings for every real group. A checker that cries wolf gets ignored.
REQUIRES_COMMENT = frozenset(
    {"STRUCT_DECL", "CLASS_DECL", "ENUM_DECL", "FUNCTION_DECL", "CXX_METHOD", "CONSTRUCTOR"}
)

# The kinds that can be = delete.
METHOD_KINDS = frozenset({"CXX_METHOD", "CONSTRUCTOR"})


def clang_arguments():
    """The command line clang is given for every header.

    Warnings are off. The bundled libclang is older than this MSVC standard library
    expects, so <vector> raises a version static_assert on every parse. That is noise
    inside the standard library rather than a problem with the header under test, and
    the declarations come out identical either way - checked against the JSON
    implementation this replaced, which found no declaration this one misses.

    Example:

        "-std=c++20" in clang_arguments()          # True
        any(a.startswith("-Isrc") for a in clang_arguments())   # True
    """
    arguments = ["-std=c++20", "-x", "c++", "-fparse-all-comments", "-w"]
    arguments.extend(f"-I{path}" for path in INCLUDE_PATHS)
    return arguments


def comment_segments(raw):
    """A raw comment split into prose paragraphs and code blocks, in order.

    Only the marker stripping is here: clang hands back raw_comment with its `//` intact,
    and what to do with the stripped lines is doc_page's, because the file index renders
    the same blocks.

    Example:

        comment_segments("// What it is." + chr(10) + "//" + chr(10) + "//     call();")
        # [('prose', 'What it is.'), ('code', 'call();')]

        comment_segments(None)   # []

    Args:
        raw: cursor.raw_comment, which is None where no comment attaches.
    """
    if not raw:
        return []
    lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        # Every marker this codebase, or a stray Doxygen block, might use.
        for marker in ("///", "//!", "//", "/**", "/*", "*/"):
            if stripped.startswith(marker):
                stripped = stripped[len(marker) :]
                break
        else:
            if stripped.startswith("*"):
                stripped = stripped[1:]
        # One space after the marker is the separator, not indentation.
        if stripped.startswith(" "):
            stripped = stripped[1:]
        lines.append(stripped.rstrip())
    return doc_page.segments_from_lines(lines)


def flatten(segments):
    """The prose of a comment as one line, for a terminal report and an emptiness test.

    Example:

        flatten([('prose', 'What it is.'), ('code', 'call();')])   # 'What it is.'
        flatten([])                                                # ''

    Args:
        segments: what comment_segments returned.
    """
    return " ".join(body for kind, body in segments if kind == "prose").strip()


def declarations_in(header, index):
    """Every listed declaration the header itself introduces, with its comment.

    Only public declarations are kept. A private or protected member is implementation,
    and a private nested type's own members are public by the struct default, so without
    the parent check the walk would climb back out of the hiding and list them.

    Example:

        found = declarations_in(Path("src/value_range.h"), cindex.Index.create())
        [entry["name"] for entry in found]
        # ['ValueRange', 'observe', 'normalize', 'isEstablished']

    Args:
        header: path to the header to parse.
        index: a clang Index, reused across headers so the library loads once.
    """
    unit = index.parse(str(header), args=clang_arguments())
    if unit is None:
        raise RuntimeError(f"libclang could not parse {header}")

    target = header.name
    found = []
    for cursor in unit.cursor.walk_preorder():
        location = cursor.location.file
        # A declaration from an include is not this header's surface.
        if location is None or Path(location.name).name != target:
            continue
        kind = cursor.kind.name
        if kind not in LISTED_KINDS or not cursor.spelling:
            continue
        # INVALID is what clang reports at namespace scope, where access does not apply.
        if cursor.access_specifier.name not in ("PUBLIC", "INVALID"):
            continue
        parent = cursor.semantic_parent
        # A hidden enclosing type takes its members with it.
        if parent is not None and parent.access_specifier.name in ("PRIVATE", "PROTECTED"):
            continue
        # A deleted member is one of a block written under a single comment - "this type
        # has no value semantics" is said once, not five times.
        deleted = kind in METHOD_KINDS and cursor.is_deleted_method()
        enclosing = ""
        if parent is not None and parent.kind.name in ("STRUCT_DECL", "CLASS_DECL"):
            enclosing = parent.spelling
        found.append(
            {
                "kind": kind,
                "name": cursor.spelling,
                "signature": cursor.type.spelling if cursor.type is not None else "",
                "segments": comment_segments(cursor.raw_comment),
                "comment": flatten(comment_segments(cursor.raw_comment)),
                "line": cursor.location.line,
                "required": kind in REQUIRES_COMMENT and not deleted,
                "deleted": deleted,
                "enclosing": enclosing,
            }
        )
    return mark_continuations(found)


def mark_continuations(found):
    """Flag each declaration that a comment above its group already covers.

    A comment over a run of constants or fields attaches to the first of them, so the
    rest arrive with no comment of their own. They are not undocumented - the page has to
    say which is which, or every group reads as a row of gaps.

    Only declarations that do not require a comment are marked. For a function a shared
    comment is a real gap, and four were found and fixed that way; swallowing those would
    turn the checker into one that cannot fire.

    Adjacency is by line, because a comment two declarations up is a group and one twenty
    lines up is a coincidence.

    Example:

        mark_continuations([
            {"name": "A", "comment": "the palette", "required": False, "line": 1},
            {"name": "B", "comment": "", "required": False, "line": 2}])[1]["continues"]
        # True

    Args:
        found: declarations in source order, each carrying its line.
    """
    covered_until = None
    for entry in found:
        entry["continues"] = False
        if entry["comment"]:
            covered_until = entry["line"]
            continue
        # Within two lines of the last commented declaration, or of one already
        # continuing it, so a blank line inside a group does not break the run.
        if not entry["required"] and covered_until is not None and entry["line"] - covered_until <= 2:
            entry["continues"] = True
            covered_until = entry["line"]
        else:
            covered_until = None
    return found


def collect(source_directory):
    """Every header paired with the declarations it introduces.

    Example:

        parsed = collect(Path("src"))
        parsed[0]["header"]              # 'az_network.h'
        len(parsed[0]["declarations"])   # 22

    Args:
        source_directory: directory of headers to parse, non-recursively.
    """
    index = cindex.Index.create()
    return [
        {"header": header.name, "declarations": declarations_in(header, index)}
        for header in sorted(source_directory.glob("*.h"))
    ]


def render(parsed):
    """The whole page, as one self-contained HTML string.

    Example:

        page = render(collect(Path("src")))
        page.startswith("<!DOCTYPE html>")   # True

    Args:
        parsed: records from collect.
    """
    total = sum(len(entry["declarations"]) for entry in parsed)
    required = [
        declaration
        for entry in parsed
        for declaration in entry["declarations"]
        if declaration["required"]
    ]
    documented = sum(1 for declaration in required if declaration["comment"])
    parts = [
        doc_page.head("src api surface", "api"),
        "<h1>src, declaration by declaration</h1>",
        f'<p class="sub">{total} declarations across {len(parsed)} headers. '
        f"{documented} of {len(required)} classes, functions and enums carry a comment; "
        "fields are listed but not required, because a comment above a group of them "
        "attaches to the first. Parsed by libclang, not by pattern matching. "
        "Generated - regenerate rather than edit.</p>",
    ]
    listed = [entry for entry in parsed if entry["declarations"]]
    parts.append('<div class="jump">')
    parts.extend(
        f'<a href="#{html.escape(entry["header"])}">{html.escape(entry["header"])}</a>'
        for entry in listed
    )
    parts.append("</div>")
    for entry in listed:
        # The id is the file name, which is unique and is what a reader would type. The
        # back link goes to the other page rather than up this one.
        parts.append(
            f'<h2 id="{html.escape(entry["header"])}">{html.escape(entry["header"])}'
            f'<a class="back" href="src_index.html">what it is for</a></h2>'
        )
        # Fixed layout: an example inside a cell would otherwise size the column to its
        # longest line and push the prose column off the page.
        parts.append('<div class="scroll"><table class="decls"><thead><tr>')
        parts.append("<th>Declaration</th><th>Type</th><th>What it is</th>")
        parts.append("</tr></thead><tbody>")
        for declaration in entry["declarations"]:
            owner = (
                f'<span class="owner">{html.escape(declaration["enclosing"])}::</span>'
                if declaration["enclosing"]
                else ""
            )
            # A deleted member is not undocumented, it is removed from the interface.
            # Saying "no comment" of one reads as an omission somebody should fix.
            if declaration["continues"]:
                empty_note = "described with the line above"
            elif declaration["deleted"]:
                empty_note = ""
            else:
                empty_note = "no comment"
            said = doc_page.render_comment(declaration["segments"], empty_note)
            if declaration["deleted"]:
                said = '<span class="tag">deleted</span>' + (said if said else "")
            parts.append(
                f"<tr><td><code>{owner}{html.escape(declaration['name'])}</code></td>"
                f'<td><code>{html.escape(declaration["signature"])}</code></td>'
                f"<td>{said}</td></tr>"
            )
        parts.append("</tbody></table></div>")
    parts.append(
        doc_page.foot(
            "Presence is all this measures. Whether a comment says anything worth "
            "reading is settled by reading it."
        )
    )
    return "\n".join(parts)


def main():
    """Parse the headers, report comment coverage, and write the page unless checking."""
    parser = argparse.ArgumentParser(description="List src/ declarations and their comments.")
    parser.add_argument("--source", default="src", help="directory of headers")
    parser.add_argument("--out", default="docs/src_api.html", help="page to write")
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    parser.add_argument("--only", default="", help="parse one header and print it")
    arguments = parser.parse_args()

    if cindex is None:
        print("libclang is not installed: pip install libclang")
        return 2

    source_directory = Path(arguments.source)
    try:
        # One header at a time when asked, so a parse failure is readable.
        if arguments.only:
            header = source_directory / arguments.only
            for declaration in declarations_in(header, cindex.Index.create()):
                missing = declaration["required"] and not declaration["comment"]
                owner = f'{declaration["enclosing"]}::' if declaration["enclosing"] else ""
                mark = "! " if missing else "  "
                print(f'{mark}{owner}{declaration["name"]:<34} {declaration["signature"]}')
            return 0
        parsed = collect(source_directory)
    except (RuntimeError, OSError) as error:
        print(error)
        return 2

    # The findings first, so they are not below a summary line that reads as a pass.
    undocumented = [
        (entry["header"], declaration)
        for entry in parsed
        for declaration in entry["declarations"]
        if declaration["required"] and not declaration["comment"]
    ]
    for header, declaration in undocumented:
        owner = f'{declaration["enclosing"]}::' if declaration["enclosing"] else ""
        print(f'  {header:<26} {owner}{declaration["name"]:<30} no comment')

    if not arguments.check:
        out_path = Path(arguments.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # CRLF, as everything in this repository is.
        out_path.write_text(render(parsed), encoding="utf-8", newline="\r\n")
        print(f"wrote {out_path}")

    total = len(
        [
            declaration
            for entry in parsed
            for declaration in entry["declarations"]
            if declaration["required"]
        ]
    )
    print(f"{total - len(undocumented)} of {total} required declarations documented, ", end="")
    print(f"{len(parsed)} headers parsed")
    return 1 if undocumented else 0


if __name__ == "__main__":
    sys.exit(main())
