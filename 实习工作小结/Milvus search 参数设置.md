# Milvus search 参数设置





**内积**
$$
IP(u, v) = ∑ (u_i * v_i)
$$


**余弦相似度**
$$
COSINE(u, v) = (u • v) / (||u|| ||v||)
$$


**L2 距离 (欧氏距离)**
$$
L2(u, v) = sqrt(∑ ((u_i - v_i)^2))
$$


**杰卡德相似系数 (JACCARD)** 
$$
JACCARD(u, v) = \frac{|u \cap v|}{|u \cup v|}
$$
 **汉明距离 (HAMMING)**

在向量语境中，汉明距离通常用于比较离散符号或编码，如二进制向量。



向量检索处理步骤：

```
# 001大模型qwen2.5 72b 提取漏洞情报CPE
# 002 CPE2Json
# 003 Json2Embedding：bge-m3->embedding
# 004 search milvus vector database
# Group search results
# res = client.search(
#     collection_name="quick_setup1",  # Collection name
#     data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
#     search_params={
#         "metric_type": "L2",
#         "params": {"nprobe": 10},
#     },  # Search parameters
#     limit=5,  # Max. number of groups to return
#     group_by_field="color",  # Group results by document ID
#     group_size=2,  # returned at most 2 passages per document, the default value is 1
#     group_strict_size=True,  # ensure every group contains exactly 3 passages
#     output_fields=["color"]
# )
# res内容：
# [
#     {
#         "id": 341,
#         "distance": 1.8011726140975952,
#         "entity": {'cpe_json':{xxx}}
#     },
#     {
#         "id": 432,
#         "distance": 1.6463967561721802,
#      "entity": {'cpe_json':{xxx}}
#     },
#     {
#         "id": 55,
#         "distance": 1.6183757781982422,
#      "entity": {'cpe_json':{xxx}}
#     },
#     {
#         "id": 749,
#         "distance": 1.6102802753448486,
#         "entity": {'cpe_json':{xxx}}
#     },
#     {
#         "id": 95,
#         "distance": 1.5998632907867432,
#        "entity": {'cpe_json':{xxx}}
#     }
# ]
# 005 show
```

{
    "cpe": "[cpe:/a:ibm:cognos_controller:10.4.0:*:*:]"
}

001 cpe2json.py  连接到clickhouse数据库，然后cpe2json，将json保存到cpe_data.json中；

然后使用batch_insert.py脚本连接mivlus 数据库，创建新集合，wjy_demo4, embedding_model = bge_m3,然后向量化到向量数据库中。

002 





在 `Milvus` 中，`client.search` 和 `client.query` 都用于从集合中获取数据，但它们的用途和功能有一些显著的不同。以下是它们的主要区别：

### 1. **`client.search`**

`client.search` 是用于 **向量检索** 的方法，主要用于基于向量相似度进行搜索。它通过与指定向量的相似度（如余弦相似度、欧几里得距离等）进行比较，从 Milvus 中检索相似的向量数据。

- **用途**：
  - 用于 **基于向量的相似度搜索**。
  - 通常用在检索最相似的项，比如图像、文本、视频等。
  - 搜索时，你需要传入一个 **查询向量** 和搜索的 **k 个最相似项**，并可以指定相关的过滤条件。
- **参数**：
  - `data`：待查询的向量数据。
  - `top_k`：返回与查询向量最相似的 `k` 个结果。
  - `params`：设置检索的距离度量方式（如 `L2` 或 `IP`）。
  - `filter`（可选）：用于根据特定的条件过滤检索结果。
  - `output_fields`（可选）：指定要返回的字段。
- **返回值**：
  - 返回最相似的 `k` 个向量，以及它们的相关信息（如 ID 和距离）。

#### 示例：

```python
results = client.search(
    data=query_vectors,  # 查询向量数据
    anns_field="embedding",  # 向量字段
    param={"metric_type": "L2", "nprobe": 10},  # 距离度量方式和搜索策略
    top_k=5,  # 返回最相似的 5 个结果
    filter="age >= 30",  # 可选的过滤条件
    output_fields=["name", "age"],  # 可选的返回字段
    collection_name="user_data"  # 集合名称
)
```

### 2. **`client.query`**

`client.query` 是用于 **基于字段的查询** 的方法，类似于传统的 SQL 查询。你可以使用该方法根据指定的条件过滤数据，并返回字段数据。这个方法更适合进行 **基于非向量字段的查询**，例如基于 ID、类别或时间等字段的查询。

- **用途**：
  - 用于 **基于属性字段的查询**，例如根据某些字段（如 ID、类型、时间等）来过滤数据。
  - 不涉及向量相似度，而是基于传统的过滤条件（比如 `==`、`<`、`>=` 等）。
- **参数**：
  - `filter`：指定过滤条件的表达式。
  - `output_fields`：指定要返回的字段。
  - `collection_name`：查询的集合名称。
- **返回值**：
  - 返回符合过滤条件的记录。

#### 示例：

```python
results = client.query(
    collection_name="user_data",  # 集合名称
    filter="age >= 30 AND gender == 'male'",  # 过滤条件
    output_fields=["name", "age", "gender"]  # 返回字段
)
```

### 区别总结：

| **特点**     | **`client.search`**                                  | **`client.query`**                              |
| ------------ | ---------------------------------------------------- | ----------------------------------------------- |
| **主要用途** | 向量检索，基于相似度查找数据                         | 基于字段查询，支持过滤字段                      |
| **输入**     | 查询向量、k 值、距离度量、过滤条件等                 | 过滤条件（基于字段）、字段输出等                |
| **输出**     | 返回与查询向量最相似的 `k` 个结果                    | 返回满足条件的记录                              |
| **适用场景** | 用于图像、文本等的相似度检索，返回最相似的项         | 用于基于字段（如年龄、性别等）的条件查询        |
| **检索方式** | 基于向量的相似度计算（如余弦相似度、欧几里得距离等） | 基于传统字段的条件过滤（如 `==`, `>`, `<=` 等） |

### 何时使用 `client.search` 和 `client.query`

- **使用 `client.search`**：当你需要从大量向量数据中找到最相似的项时，比如在图像检索、自然语言处理（如文本检索）、推荐系统中，通常会使用 `client.search`。
- **使用 `client.query`**：当你需要基于某些标准字段进行条件查询时，比如在 SQL 数据库中的常规数据过滤操作，适合使用 `client.query`。