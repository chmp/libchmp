import argparse
import pathlib
import subprocess
import sys

self_path = pathlib.Path(__file__).parent.resolve()


def cmd():
    def decorator(func):
        func.__chmp__ = getattr(func, "__chmp__", {})
        func.__chmp__["name"] = func.__name__
        return func

    return decorator


@cmd()
def precommit():
    format()
    test()
    doc()


@cmd()
def format():
    run(sys.executable, "-m", "black", self_path)


@cmd()
def test():
    run(sys.executable, "-m", "pytest", self_path)


@cmd()
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
        if not callable(func) or not hasattr(func, "__chmp__"):
            continue

        subparser = subparsers.add_parser(func.__chmp__["name"])
        subparser.set_defaults(__main__=func)

    return parser


if __name__ == "__main__":
    main()
