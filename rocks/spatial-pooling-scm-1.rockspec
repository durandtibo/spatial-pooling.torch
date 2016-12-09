package = "spatial-pooling"
version = "scm-1"

source = {
   url = "https://github.com/durandtibo/spatial-pooling.torch.git",
   tag = "master"
}

description = {
   summary = "Spatial pooling for Torch7 nn",
   detailed = [[
Torch7 Implementation of different spatial pooling.
   ]],
   homepage = "https://github.com/durandtibo/spatial-pooling.torch.git",
   license = "MIT License"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0"
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)"
   }
}
