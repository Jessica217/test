def search_cpe(input_text):
    try:
        # 生成用户消息并调用大模型
        user_message = get_prompt(input_text)
        model_name = 'qwen2.5:72b'
        content = chat_with_qwen_72b(client, model_name, user_message)
        print('content====================', content)

        # 提取第一个 CPE
        first_cpe = extract_all_cpe_strings(content)
        print(type(first_cpe))
        if not first_cpe:
            return {"cpe": "No valid CPE extracted."}, pd.DataFrame()
        print(first_cpe) #['cpe:/a:adobe:acrobat:*:*:*:*', 'cpe:/a:adobe:reader:*:*:*:*']大模型返回两个数据

        # 转为 JSON 格式
        cpe_json = cpe_to_json(first_cpe)
        # cpe_json_formatted = json.dumps({"cpe": first_cpe, "json": cpe_json}, indent=4)
        cpe_json_formatted = json.dumps({"cpe": first_cpe}, indent=4)


        # 使用嵌入模型生成查询向量
        doc_embedding = bge_m3_ef.encode_queries([json.dumps(cpe_json)])
        dense_vector = doc_embedding['dense'][0]

        # 搜索 Milvus 数据库
        search_params = {
            "metric_type": "COSINE",
            "params": {'nprobe': 20, # 在每一层扩展20个候选节点
            'level': 3,
            'radius': 0.8, # 外边界
            'range_filter': 1 }
        }
        results = collection.search(
            data=[dense_vector], anns_field="vector", param=search_params,
            limit=20, output_fields=["part", "vendor", "product", "version", "update", "edition", "language"]
        )

        # 格式化结果
        output_data = []
        index = 1
        for hits in results:
            for hit in hits:
                row = {
                    "index": index,
                    "part": hit.entity.get("part"),
                    "vendor": hit.entity.get("vendor"),
                    "product": hit.entity.get("product"),
                    "version": hit.entity.get("version"),
                    "update": hit.entity.get("update"),
                    "edition": hit.entity.get("edition"),
                    "language": hit.entity.get("language"),
                    "score": round(hit.score * 100, 2)
                }
                output_data.append(row)
                index += 1

        # 转换为 DataFrame
        df = pd.DataFrame(output_data)
        return cpe_json_formatted, df
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()