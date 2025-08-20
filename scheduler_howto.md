How to use the scheduler

Auto-start at launch
./training-camp serve \
  -m latin.model.gob -p 8080 \
  --auto-task "translate english text to latin" \
  --auto-count 25 \
  --auto-every 15m

Control via API

Start:

curl -X POST http://localhost:8080/scheduler/start \
  -H "Content-Type: application/json" \
  -d '{"task":"translate english text to latin","count":50,"every":"30m"}'


Status:

curl http://localhost:8080/scheduler/status


Stop:

curl -X POST http://localhost:8080/scheduler/stop


This will steadily fetch fresh examples from OpenAI, embed, append to the in-memory KNN, and persist updates to your model.gob.
