module.exports = {
  run: [{
    method: "shell.run",
    params: {
      path: ".",
      message: "git pull"
    }
  }, {
    method: "script.start",
    params: {
      uri: "install.js"
    }
  }]
}
