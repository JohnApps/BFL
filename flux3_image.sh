
curl -X POST 'https://api.bfl.ai/v1/flux-3-image' \
  -H 'Content-Type: application/json' \
  -H 'X-Key: $BFL_API_KEY' \
  -d '{
  "prompt": "tdarken the image and make the colors more intense\n",
  "aspect_ratio": "auto",
  "resolution": "1k",
  "grounding": true,
  "safety_tolerance": 2
}'