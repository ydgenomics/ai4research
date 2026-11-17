curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_ogr/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" -H "Content-Type: application/json" \
  -d '{
    "model":"OGR",
    "model_name":"1B_8k",
    "mode":"predict",
    "sequence":"ACGTTGCATGCAACGT",
    "predict_length":10
}'

curl -X POST "https://www.dcs.cloud/api/aigress/openai/rice_ogr" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -d '{"model": "OGR2", "model_name": "1B_8k", "mode": "predict", "sequence": "ACGTTGCATGCAACGT", "predict_length": 5}'