from chmp.torch_utils._aop import (
    joinpoint,
    System,
    proceed,
    replace,
    decorate,
)


def test_example():
    assert system(2) == 4


def test_customized():
    new_system = system.copy()

    @decorate(new_system, "inner")
    def _(val):
        return -proceed(val)

    assert new_system(2) == -4


def test_multiple_proceeds():
    new_system = system.copy()

    @decorate(new_system, "inner")
    def _(val):
        return -proceed(val)

    @decorate(new_system, "inner")
    def _(val):
        return proceed(val) + proceed(val)

    assert new_system(2) == -8


system = System("root")


@replace(system, "root", prototype=True)
def root(val):
    return joinpoint("inner")(val)


@replace(system, "inner", prototype=True)
def inner(val):
    return 2 * val
