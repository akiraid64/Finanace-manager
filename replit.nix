{pkgs}: {
  deps = [
    pkgs.gcc
    pkgs.cmake
    pkgs.libxcrypt
    pkgs.eigen
    pkgs.catch
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.cairo
    pkgs.glibcLocales
    pkgs.postgresql
    pkgs.openssl
  ];
}
