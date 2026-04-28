from pyspark import SparkContext, SparkConf

# 初始化 SparkContext（适配你的 Spark 4.1.1）
conf = SparkConf().setAppName("Exp3_Ultimate").setMaster("local[*]")
sc = SparkContext(conf=conf)

print("=" * 50)
print("          Spark RDD 编程基础实验（带容错处理）")
print("=" * 50)

# -------------------------- 实验1：部门数据解析 --------------------------
print("\n=== 实验1：读取并解析 departments.txt ===")
dept_rdd = sc.textFile("departments.txt")
parsed_dept = dept_rdd.map(lambda line: line.strip().split(","))
# 过滤脏数据（必须有2个字段）
parsed_dept = parsed_dept.filter(lambda x: len(x) == 2 and x[0] and x[1])
print("部门数据解析结果（前5条）：")
for item in parsed_dept.collect():
    print(f"部门编号: {item[0]}, 部门名称: {item[1]}")

# -------------------------- 实验2：员工数据与部门关联（带容错） --------------------------
print("\n=== 实验2：员工数据与部门关联查询 ===")
emp_rdd = sc.textFile("employees.txt")

# 1. 原始数据
print("employees.txt 原始数据前10行：")
print(emp_rdd.take(10))

# 2. 解析并过滤脏数据（必须有3个字段，且工资是数字）
def parse_and_filter_emp(line):
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) != 3:
        return None
    dept_id, name, salary_str = parts
    try:
        salary = int(salary_str)
        return (dept_id, (name, salary))
    except ValueError:
        return None

emp_kv = emp_rdd.map(parse_and_filter_emp).filter(lambda x: x is not None)
print("过滤后的员工数据前10条：")
print(emp_kv.take(10))

# 3. 部门数据转换为 (部门号, 部门名称)
dept_kv = parsed_dept.map(lambda x: (x[0], x[1]))
print("部门数据前5条：")
print(dept_kv.take(5))

# 4. 内连接
joined = emp_kv.join(dept_kv)
print("员工-部门关联结果（前10条）：")
for item in joined.take(10):
    dept_id = item[0]
    emp_info = item[1][0]
    dept_name = item[1][1]
    print(f"部门: {dept_name}({dept_id}), 姓名: {emp_info[0]}, 工资: {emp_info[1]}")

# -------------------------- 实验3：按部门统计平均工资和人数 --------------------------
print("\n=== 实验3：按部门统计员工人数与平均工资 ===")
if joined.isEmpty():
    print("⚠️  没有关联数据，跳过统计")
else:
    dept_stats = joined.map(lambda x: (x[1][1], (x[1][0][1], 1))) \
        .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
        .map(lambda x: (x[0], x[1][1], round(x[1][0]/x[1][1], 2)))
    print("部门统计结果：")
    for row in dept_stats.collect():
        print(f"部门: {row[0]}, 员工数: {row[1]}, 平均工资: {row[2]}")

# -------------------------- 实验4：WordCount（sentences.txt） --------------------------
print("\n=== 实验4：sentences.txt 词频统计 ===")
sent_rdd = sc.textFile("sentences.txt")
words = sent_rdd.flatMap(lambda line: line.strip().split())
word_counts = words.filter(lambda w: w).map(lambda w: (w.lower(), 1)).reduceByKey(lambda a,b: a+b)
top_words = word_counts.sortBy(lambda x: x[1], ascending=False).take(10)
print("出现频率最高的前10个单词：")
for word, count in top_words:
    print(f"{word}: {count}次")

sc.stop()
print("\n✅ 所有实验步骤执行完成！")