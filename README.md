# Deploy Wav2vec2 for ASR with FastAPI

Deploy a pre-trained Wav2vec2 model for ASR as a REST API using FastAPI

## Demo

The model is trained to trascribe speech into text  on a thai common voice (8th version) dataset. Here's a sample request to the API:

```bash
http POST http://127.0.0.1:8000/predict audio_path="<path to audio>"
```

The response you'll get looks something like this:

```js
{
    "transcription": "วัน นี้ กิน อะไร ดี",
}
```
The original tutorial is deploy bert for sentiment analysis as rest api using pytorch transformers by hugging face and fast API.
You can also [read the complete tutorial here](https://www.curiousily.com/posts/deploy-bert-for-sentiment-analysis-as-rest-api-using-pytorch-transformers-by-hugging-face-and-fastapi/).

## Installation

Clone this repo:

```sh
git clone https://github.com/ArtitSu/asr-api
```

Install the dependencies:

```sh
pip install fastapi uvicorn pydantic
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install https://github.com/kpu/kenlm/archive/master.zip pyctcdecode
apt-get install httpie
```

## Test the setup

Start the HTTP server:

```sh
bin/start_server
```

Send a test request:

```sh
bin/test_request
```

## License
