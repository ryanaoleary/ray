load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

exports_files(
    [
        "redis-server.exe",
        "redis-cli.exe",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_srcs",
    srcs = glob(
        include = ["**"],
        exclude = ["*.bazel"],
    ),
)


make(
    name = "redis",
    args = [
        "BUILD_TLS=yes",
        "-s",
    ],
    copts = [
        "-DLUA_USE_MKSTEMP",
        "-Wno-pragmas",
        "-Wno-empty-body",
        "-fPIC",
    ],
    visibility = ["//visibility:public"],
    lib_source = ":all_srcs",
    deps = [
        "@openssl//:openssl",
    ],
    out_binaries = [
        "redis-server",
        "redis-cli"
    ]
)

genrule_cmd = select({
    "@platforms//os:osx": """
        unset CC LDFLAGS CXX CXXFLAGS
        tmpdir="redis.tmp"
        p=$(location Makefile)
        cp -p -L -R -- "$${p%/*}" "$${tmpdir}"
        chmod +x "$${tmpdir}"/deps/jemalloc/configure
        parallel="$$(getconf _NPROCESSORS_ONLN || echo 1)"
        make -s -C "$${tmpdir}" -j"$${parallel}" V=0 CFLAGS="$${CFLAGS-} -DLUA_USE_MKSTEMP -Wno-pragmas -Wno-empty-body"
        mv "$${tmpdir}"/src/redis-server $(location redis-server)
        chmod +x $(location redis-server)
        mv "$${tmpdir}"/src/redis-cli $(location redis-cli)
        chmod +x $(location redis-cli)
        rm -r -f -- "$${tmpdir}"
    """,
    "//conditions:default": """
        cp $(RULEDIR)/redis/bin/redis-server $(location redis-server)
        cp $(RULEDIR)/redis/bin/redis-cli $(location redis-cli)
    """
})

genrule_srcs = select({
    "@platforms//os:osx": glob(["**"]),
    "//conditions:default": [":redis"],
})


genrule(
    name = "bin",
    srcs = genrule_srcs,
    outs = [
        "redis-server",
        "redis-cli",
    ],
    cmd = genrule_cmd,
    visibility = ["//visibility:public"],
    tags = ["local"],
)
