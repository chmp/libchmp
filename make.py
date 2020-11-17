import argparse
import pathlib
import subprocess
import sys

self_path = pathlib.Path(__file__).parent.resolve()

_md = lambda effect: lambda f: [f, effect(f)][0]
_ps = lambda o: vars(o).setdefault("__chmp__", {})
cmd = lambda **kw: _md(lambda f: _ps(f).update(kw))
arg = lambda *a, **k: _md(lambda f: _ps(f).setdefault("__args__", []).append((a, k)))


@cmd(help="Perform all important tasks before a commit")
def precommit():
    format()
    test()
    doc()


@cmd(help="Format the source code")
def format():
    run(sys.executable, "-m", "black", self_path)


@cmd(help="Run unittests")
@arg("--pdb", action="store_true", default=False)
def test(pdb=False):
    run(
        sys.executable,
        "-m",
        "pytest",
        self_path,
        *(["--pdb"] if pdb else []),
    )


@cmd(help="Update the documentation")
def doc():
    from chmp.tools.mddocs import transform_directories

    transform_directories(self_path / "docs" / "src", self_path / "docs")


def run(*args, **kwargs):
    args = [str(arg) for arg in args]
    print("::", " ".join(args))
    return subprocess.run(args, **kwargs)


def main():
    parser = _build_parser()
    args = vars(parser.parse_args())

    if "__main__" not in args:
        return parser.print_help()

    func = args.pop("__main__")
    return func(**args)


def _build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    for func in globals().values():
        if not hasattr(func, "__chmp__"):
            continue

        desc = dict(func.__chmp__)
        name = desc.pop("name", func.__name__)
        args = desc.pop("__args__", [])

        subparser = subparsers.add_parser(name, **desc)
        subparser.set_defaults(__main__=func)

        for arg_args, arg_kwargs in args:
            subparser.add_argument(*arg_args, **arg_kwargs)

    return parser


if __name__ == "__main__":
    main()
