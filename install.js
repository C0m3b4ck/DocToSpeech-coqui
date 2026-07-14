module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install -r requirements-coqui.txt",
          "uv pip install -r requirements-styletts2.txt",
          "uv pip install gradio"
        ],
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "."
        }
      }
    },
    {
      method: 'input',
      params: {
        title: "Install Complete",
      }
    },
  ]
}
