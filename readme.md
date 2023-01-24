<br>
<p align="center">
  <img src="images/logo.png" width="150">
</p>

Parrot Inference Server
=======================
This is the inference server for the Parrot autocomplete plugin (VSCode/IntelliJ). 


### 🔌 Plugins for IDEs
+ [Parrot for IntellJ](https://github.com/mamiksik/parrot-intellij)
+ [Parrot for VSCode](https://github.com/mamiksik/parrot-vscode)

### 🦾 Trained Models
+ [Parrot commit generation Model](https://huggingface.co/mamiksik/CommitPredictorT5PL)
+ [Parrot autocomplete model](https://huggingface.co/mamiksik/CodeBERTa-commit-message-autocomplete)

### 📚 Others
- [Parrot Dataset](https://huggingface.co/datasets/mamiksik/processed-commit-diffs)
- [Parrot Inference Server](https://github.com/mamiksik/ParrotInferenceServer)


Install dependencies using poetry and then run the server using uvicorn.
```bash
$ poetry install
$ poetry run uvicorn src.main:app
```

The server will most likely be available at http://127.0.0.1:8000. 

