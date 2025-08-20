Training Camp

A Go-based CLI and API to generate, embed, train, serve, and continually refine narrow AI models using OpenAI embeddings and a lightweight KNN-style classifier — no Python required.

Features

Prompt-based training data generation via OpenAI

Embedding via OpenAI’s text-embedding-3-small or text-embedding-3-large

Pure Go 1-NN/KNN classifier (serializable to .gob)

Accuracy evaluation tools

HTTP API server with prediction + batch prediction endpoints

Minimal frontend UI served at /

Configurable background scheduler for "continuing education" (periodic retraining with new samples)

Install

cd training-camp
go build -o training-camp


Set your OpenAI key:

export OPENAI_API_KEY="sk-..."


(Or place it in a .env file which the app can load at runtime.)

Quickstart (Latin Translation Example)
# 1. Generate training examples
./training-camp generate --task "translate english text to latin" --count 10 -o latin.json

# 2. Embed them with OpenAI
./training-camp embed -i latin.json -o latin.embeddings.json

# 3. Train a classifier
./training-camp train -i latin.embeddings.json -o latin.model.gob

# 4. Evaluate on test data
./training-camp eval -m latin.model.gob -i latin.embeddings.json

# 5. Serve predictions
./training-camp serve -m latin.model.gob -p 8080

Serving API
Single Prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.12, 0.34, 0.56]}'


Response:

{"label":"latin"}

Batch Prediction
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"embeddings":[[0.12,0.34,0.56],[0.98,0.76,0.54]]}'


Response:

{"labels":["latin","latin"]}

Text Classification (frontend + API)

Send plain text and let the server embed + classify:

curl -X POST http://localhost:8080/classify_text \
  -H "Content-Type: application/json" \
  -d '{"text":"I love you"}'


Response:

{"label":"Te amo"}


Open http://localhost:8080/ in a browser to try the built-in UI.

Continuing Education (Background Scheduler)

Training Camp can periodically “go back to the well” by generating new training data from OpenAI, embedding it, and retraining the model. This allows the model to evolve over time with fresh examples.

Enable in your task config:

task: "translate english text to latin"
count: 20
schedule_minutes: 60
output: "latin.json"


When schedule_minutes is set, the server runs a background loop:

Generate new examples

Embed them

Retrain and hot-swap the model

This allows narrow AI models to improve incrementally without manual retraining.

Project Layout
training-camp/
├── cmd/              # CLI commands
├── internal/         # core logic (classifier, embeddings, scheduler, etc.)
├── ui/               # static HTML served at /
├── go.mod
└── README.md

Use Cases

Latin translation classifier

Watch authenticity detection

Jira incident classifier: detect noisy alerts, categorize incidents, and extrapolate metrics like time-to-detect/mitigate/resolve

Roadmap

X OpenAI generation + embedding

X Offline training & eval

X Serve predictions via HTTP

X Minimal UI frontend

X Background scheduler for continuing education

 Multi-label classification

 Export predictions in batch

 Config-based fine-tuning presets

License

MIT
