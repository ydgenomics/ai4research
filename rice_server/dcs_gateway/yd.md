# 冒烟调用（DCS 平台，经统一网关）

> 完整示例见 [API.md](API.md)。OGR-Base 已无 `model_name` / `mode`：模型固定 `1B_8k`，功能由子路径决定。

```bash
dcs_host="https://www.dcs.cloud"
dcs_entry="/api/aigress/openai/OGR"
dcs_api_key="<YOUR_API_KEY>"

# OGR-Base：NTP 下游碱基预测
curl -X POST ${dcs_host}${dcs_entry}/OGR-Base/NTP \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{"sequence":"ACGTTGCATGCAACGT","predict_length":10}'

# OGR-Base：Embedding 提取
curl -X POST ${dcs_host}${dcs_entry}/OGR-Base/Embedding \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{"sequence":"ACGTTGCATGCAACGT","pooling_method":"mean"}'

# 兼容：model_sub 缺省即 OGR-Base
curl -X POST ${dcs_host}${dcs_entry}/Embedding \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{"sequence":"ACGTTGCATGCAACGT","pooling_method":"mean"}'

# 聚合健康检查（免鉴权）
curl -s ${dcs_host}${dcs_entry}/health | python3 -m json.tool
```