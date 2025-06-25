import pandas as pd
from tabulate import tabulate
from io import StringIO

# 模拟 CSV 数据（你可以粘贴完整内容）
import pandas as pd
from tabulate import tabulate
# 正确读取 CSV 文件
df = pd.read_csv("table_2/Open_overlapping_galaxies.csv")
print("解混数量：{}".format(len(df)))
# 打印 Markdown 格式的前 7 行表格
print(tabulate(df.head(10), headers='keys', tablefmt='github', showindex=False))



