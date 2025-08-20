kTraining Camp

A Go-based CLI to generate, embed, train, serve, and evaluate narrow AI models using OpenAI embeddings and KNN-style logic — no Python required.

Features

Prompt-based training data generation via OpenAI

Embedding via OpenAI's text-embedding-ada-002

Custom 1-NN classifier in pure Go (serializable)

Accuracy evaluation tools

Offline prediction API with training-camp serve

Task-based config YAMLs for reusable pipelines

Install

git clone https://github.com/your-org/training-camp.git
cd training-camp
go build -o training-camp


Export your OpenAI API key:
export OPENAI_API_KEY=sk-xxxxx

(You can also put it in a .env file.)

Quickstart (Latin Translation Example)
1. Generate training examples
./training-camp generate --task "translate english text to latin" --count 10 -o latin.json

latin.json will contain labeled training pairs.

2. Create embeddings
./training-camp embed -i latin.json -o latin.embeddings.json


3. Train a classifier
./training-camp train -i latin.embeddings.json -o latin.model.gob

3. Train a classifier

4. Evaluate on test data

If you have a latin_test.embeddings.json file:

./training-camp eval -m latin.model.gob -i latin_test.embeddings.json

Serve Predictions
./training-camp serve -m latin.model.gob -p 8080

Serving API
Single Prediction

Send one embedding vector:

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.12, 0.34, 0.56, 0.78]
  }'

{"label":"latin"}

Batch Prediction

Send multiple embeddings at once:

curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [
      [0.12, 0.34, 0.56, 0.78],
      [0.98, 0.76, 0.54, 0.32]
    ]
  }'

Response:
{
  "labels": ["latin", "latin"]
}

Browser Example (CORS Enabled)


fetch("http://localhost:8080/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ embedding: [0.12, 0.34, 0.56, 0.78] })
})
.then(res => res.json())
.then(console.log)


Hugging Face Notes

You do NOT need Hugging Face unless you want to:

Fine-tune a transformer model (Python + PyTorch)

Use pre-trained HF models for embeddings

Export or host on the Hugging Face Hub

If needed:

pip install sentence-transformers
python -m onnxruntime_tools.convert --model_path ./model --output_path ./model.onnx


Then use onnx-go for inference.

Roadmap

 OpenAI generation + embedding

 Offline training & eval

 Serve predictions via HTTP

 Batch prediction endpoint

 CORS support

 Multi-label classification

 Export predictions in batch

 Config-based fine-tuning presets

License

MIT
