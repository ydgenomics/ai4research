curl --location 'https://test.dcs.cloud/api/aigress/openai/<PATH>
--header 'Authorization: Bearer <YOUR_API_KEY>' \
--header 'Content-Type: application/json' \
--data '{
    "model": "<MODEL>"
}'


运行命令示例：

执行服务脚本时，脚本路径应为容器内部挂载后的路径。

例如：源文件 /files/dna_embedding.py 挂载到容器内 /code/dna_embedding.py，则执行命令如下：

python /code/dna_embedding.py <your_script_args>

返回结构规范：

服务脚本需返回一个 JSON 对象，其中：

必填字段：
    usage.prompt_tokens — 输入 token 数，用于计费
    usage.completion_tokens — 输出 token 数，用于计费

其余字段可根据服务需求自定义返回。

以下为示例返回结构：
    return json({
        "usage": {
            "prompt_tokens": len(sequence),
            "completion_tokens": model_token,
        },
        "status": 200,
        "message": "客户端序列 embedding 提取成功",
        "result": result,
    })
