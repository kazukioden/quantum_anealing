from pyqubo import Binary, LogEncInteger, OrderEncInteger, OneHotEncInteger, UnaryEncInteger

# Integerクラスのテスト
print("=== Integer Classes Test ===")

# LogEncodedInteger（対数エンコーディング）- メモリ効率的
y = LogEncInteger('y', (0, 7))
print(f"LogEncInteger y (0-7): {y}")

# OrderEncodedInteger（順序エンコーディング）
z = OrderEncInteger('z', (0, 4))
print(f"OrderEncInteger z (0-4): {z}")

# OneHotEncodedInteger（ワンホットエンコーディング）
w = OneHotEncInteger('w', (1, 3))
print(f"OneHotEncInteger w (1-3): {w}")

# UnaryEncodedInteger（単項エンコーディング）
u = UnaryEncInteger('u', (0, 5))
print(f"UnaryEncInteger u (0-5): {u}")

# 簡単な最適化問題の例
obj = (w - 2)**2
model = obj.compile()
qubo, offset = model.to_qubo()
print(f"\nQUBO variables: {len(qubo)}")
print(f"Offset: {offset}")