Parrot Inference Server
=======================
Run Parrot AI inference locally.


Install dependencies using poetry and then run the server using uvicorn.
```bash
$ poetry install
$ poetry run uvicorn main:app
```

The server will be available at http://127.0.0.1:8000. 
Since the local api is not protected, you don't have to set API token in the Parrot AI plugin. 

