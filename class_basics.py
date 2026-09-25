"""파이썬 class 문법 연습 예제. 위에서부터 한 단계씩 읽고 실행해보세요.

실행:
    python class_basics.py
"""
import torch
import torch.nn as nn
import random


# ---------------------------------------------------------------------------
# 예제 1. 기본 클래스: __init__, self, 메서드
# ---------------------------------------------------------------------------
class Counter:
    def __init__(self, start):
        self.value = start          # 멤버 변수는 대입하는 순간 생김

    def add(self, n):
        self.value += n             # 멤버 접근은 반드시 self.
        return self.value


def example1():
    print("== 예제 1: 기본 클래스 ==")
    c = Counter(10)                 # new 없이 함수처럼 호출
    print(c.add(5))                 # 15
    print(c.add(5))                 # 20 (상태가 유지됨)
    d = Counter(0)                  # 객체는 각자 자기 상태를 가짐
    print(c.value, d.value)         # 20 0


# ---------------------------------------------------------------------------
# 예제 2. __call__: 객체를 함수처럼 쓰기 (model(x)의 원리)
# ---------------------------------------------------------------------------
class Doubler:
    def __call__(self, x):
        return x * 2


def example2():
    print("\n== 예제 2: __call__ ==")
    f = Doubler()
    print(f(21))                    # f.__call__(21) 과 같음 -> 42


# ---------------------------------------------------------------------------
# 예제 3. 상속과 super().__init__()
# ---------------------------------------------------------------------------
class Animal:
    def __init__(self, name):
        print(f"  Animal.__init__ 실행: {name}")
        self.name = name

    def speak(self):
        return "..."


class Dog(Animal):                  # 부모는 Animal
    def __init__(self, name, breed):
        super().__init__(name)      # 부모 초기화를 직접 호출
        self.breed = breed

    def speak(self):                # 부모 메서드를 덮어씀 (override)
        return "멍멍"


def example3():
    print("\n== 예제 3: 상속 ==")
    d = Dog("보리", "푸들")
    print(d.name, d.breed, d.speak())          # 보리 푸들 멍멍
    print(isinstance(d, Animal))               # True
    print(Dog.__mro__)                         # 메서드 탐색 순서


# ---------------------------------------------------------------------------
# 예제 4. 조립(has-a): 다른 클래스 객체를 멤버로 가지기
# ---------------------------------------------------------------------------
class Engine:
    def start(self):
        return "부릉"


class Car:
    def __init__(self):
        self.engine = Engine()      # Engine을 부품으로 가짐

    def start(self):
        return "Car: " + self.engine.start()


def example4():
    print("\n== 예제 4: 조립 ==")
    print(Car().start())


# ---------------------------------------------------------------------------
# 예제 5. nn.Module 상속: 트랜스포머 코드와 같은 패턴
# ---------------------------------------------------------------------------
class TinyBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()                      # 이걸 먼저! (없으면 에러)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.linear(x))    # residual + norm


class TinyModel(nn.Module):
    def __init__(self, vocab, d_model, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab)

    def forward(self, ids):
        x = self.embed(ids)                     # (B, L) -> (B, L, d)
        for block in self.blocks:
            x = block(x)                        # block.forward(x)가 불림
        return self.out(x)                      # (B, L, vocab)


def example5():
    print("\n== 예제 5: nn.Module ==")
    model = TinyModel(vocab=10, d_model=8, n_layers=2)
    ids = torch.tensor([[1, 2, 3, 4]])
    logits = model(ids)                         # model.forward(ids)가 불림
    print("logits shape:", logits.shape)        # (1, 4, 10)

    n_params = sum(p.numel() for p in model.parameters())
    print("파라미터 수:", n_params)             # 하위 모듈 것까지 전부 모임
    print("등록된 이름 일부:", [n for n, _ in model.named_parameters()][:4])



# ---------------------------------------------------------------------------
# 예제 7
# ---------------------------------------------------------------------------
def make_batch(n):
    data = torch.zeros(n,8,dtype=torch.long)
    label = torch.zeros(n,dtype=torch.long)
    for i in range(n):
        L = random.randint(3,8)
        for j in range(L):
            data[i,j] = random.randint(1,9)
            if data[i,j] == 7: label[i] = 1
    
    return [data, label]
        
class MaskedPoolClassifier(nn.Module):
    def __init__(self, vocab, d_model, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, n_classes)
        
    def forward(self, ids):
        x = self.embed(ids)
        
        batch, length = ids.shape
        pad = torch.zeros(batch, length, dtype=torch.bool)
        pad = (ids != 0)
        
        m = pad.unsqueeze(-1)
        s = (x * m).sum(1)
        n = m.sum(1)
        mean = s / n
        
        return self.linear2(self.relu(self.linear1(mean)))
        
        



if __name__ == "__main__":
    
    model = MaskedPoolClassifier(vocab = 10, d_model = 16, n_classes = 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    
    for step in range(200):
        data, label = make_batch(64)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, label)
        if step%10==0: print(loss.item())
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        data, label = make_batch(1000)
        print(label)
        logits = model(data)
        pred = logits.argmax(-1)
        acc = (pred == label).float().mean()
    print(acc.item())