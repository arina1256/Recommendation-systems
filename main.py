import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import List, Tuple, Dict



def load_data(filepath: str) -> List[List[int]]:
    sessions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    print(f"Загружено сессий: {len(sessions)}")
    return sessions

def analyze_data(sessions: List[List[int]]):
    num_sessions = len(sessions)
    all_items = [item for session in sessions for item in session]
    unique_items = set(all_items)
    num_unique_items = len(unique_items)
    session_lengths = [len(session) for session in sessions]
    item_freq = Counter(all_items)
    
    print("\n" )
    print("Анализ данных")
    print("\n" )
    print(f"Всего сессий: {num_sessions:,}")
    print(f"Всего просмотров: {len(all_items):,}")
    print(f"Уникальных товаров: {num_unique_items:,}")
    print(f"Средняя длина сессии: {np.mean(session_lengths):.2f}")
    print(f"Медианная длина сессии: {np.median(session_lengths):.2f}")
    print(f"Минимальная и максимальная длина сессии: {min(session_lengths)} / {max(session_lengths)}")
    
    freq_values = list(item_freq.values())
    print(f"\nЧастоты товаров:")
    print(f"  Средняя: {np.mean(freq_values):.2f}")
    print(f"  Медиана: {np.median(freq_values):.2f}")
    print(f"  Топ-5 товаров: {item_freq.most_common(5)}")
    
    transitions = []
    for session in sessions:
        for i in range(len(session) - 1):
            transitions.append((session[i], session[i+1]))
    
    unique_transitions = set(transitions)
    print(f"\nПереходы:")
    print(f"  Всего переходов: {len(transitions):,}")
    print(f"  Уникальных пар: {len(unique_transitions):,}")
    print(f"  Доля уникальных: {len(unique_transitions)/len(transitions)*100:.2f}%")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(session_lengths, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Длина сессии')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].set_title('Распределение длин сессий')
    axes[0, 0].axvline(np.mean(session_lengths), color='red', linestyle='--', label=f'Среднее = {np.mean(session_lengths):.1f}')
    axes[0, 0].axvline(np.median(session_lengths), color='green', linestyle='--', label=f'Медиана = {np.median(session_lengths):.1f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    freq_sorted = sorted(freq_values, reverse=True)
    top_n = min(50, len(freq_sorted))
    axes[0, 1].plot(range(1, top_n+1), freq_sorted[:top_n], 'o-', markersize=4, color='pink')
    axes[0, 1].set_xlabel('Ранг товара')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].set_title('Топ-50 самых частых товаров')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    out_degree = Counter([t[0] for t in transitions])
    out_degree_vals = list(out_degree.values())
    
    axes[1, 0].hist(out_degree_vals, bins=30, edgecolor='black', alpha=0.7, color='cyan')
    axes[1, 0].set_xlabel('Количество уникальных переходов от товара')
    axes[1, 0].set_ylabel('Количество товаров')
    axes[1, 0].set_title('Распределение исходящих степеней')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    observations = []
    
    repeats = sum(1 for a, b in transitions if a == b)
    if repeats > 0:
        observations.append(f"Найдено {repeats} переходов из одного товара к тому же, что и был({repeats/len(transitions)*100:.2f}%)")
    
    long_sessions = [s for s in session_lengths if s > 100]
    if long_sessions:
        observations.append(f"Есть {len(long_sessions)} очень длинных сессий (>100 товаров)")
    
    gini = 1 - sum([(f/len(all_items))**2 for f in freq_values]) if freq_values else 0
    observations.append(f"Индекс Джини: {gini:.3f} ")
    
    text = "\n".join(observations) if observations else "Необычных паттернов не обнаружено"
    axes[1, 1].text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Анализ E-commerce данных', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return unique_items, item_freq



def train_test_split(sessions: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
    train_sessions = [session[:-1] for session in sessions]
    test_targets = [session[-1] for session in sessions]
    return train_sessions, test_targets



class TransitionMatrixModel:
    def __init__(self, alpha: float = 0.1):
        
        self.alpha = alpha
        self.item_to_idx = {}     
        self.idx_to_item = {}      
        self.transition_matrix = None  
        self.raw_counts = None      
        self.item_popularity = None  
        
    def build_from_sessions(self, train_sessions: List[List[int]]):
        
        print("\n" )
        unique_items = set()
        for session in train_sessions:
            unique_items.update(session)
        
        n_items = len(unique_items)
        print(f"Уникальных товаров: {n_items}")
        print(f"Теоретический размер матрицы: {n_items} x {n_items} = {n_items**2:,} ячеек")
        
        sorted_items = sorted(unique_items)
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.raw_counts = np.zeros((n_items, n_items), dtype=np.float32)
        
        total_transitions = 0
        for session in train_sessions:
            for i in range(len(session) - 1):
                current_idx = self.item_to_idx[session[i]]
                next_idx = self.item_to_idx[session[i+1]]
                self.raw_counts[current_idx, next_idx] += 1
                total_transitions += 1
        
        print(f"Всего переходов: {total_transitions:,}")
        print(f"Заполненость матрицы: {np.count_nonzero(self.raw_counts)/n_items**2*100:.2f}%")
        
        row_sums = self.raw_counts.sum(axis=1, keepdims=True)
        
        row_sums_with_smooth = row_sums + self.alpha * n_items
        
        self.transition_matrix = (self.raw_counts + self.alpha) / row_sums_with_smooth
        row_sum_check = self.transition_matrix.sum(axis=1)
        print(f"Проверка нормализации: средняя сумма строк = {row_sum_check.mean():.6f} ")
        self.item_popularity = self.raw_counts.sum(axis=0)
        print("\nТоп-5 самых вероятных переходов:")
        flat_indices = np.argsort(self.transition_matrix.flatten())[-5:][::-1]
        for flat_idx in flat_indices[:5]:
            i, j = divmod(flat_idx, n_items)
            if i != j:  
                prob = self.transition_matrix[i, j]
                from_item = self.idx_to_item[i]
                to_item = self.idx_to_item[j]
                print(f"  {from_item} -> {to_item}: {prob:.4f} ({prob*100:.2f}%)")
        
    def get_recommendations(self, last_item: int, top_k: int = 10) -> List[int]:
        
        if last_item not in self.item_to_idx:
            return self._get_fallback_recommendations(last_item, top_k)
        idx = self.item_to_idx[last_item]
        probs = self.transition_matrix[idx].copy()
        probs[idx] = 0
        top_indices = np.argsort(probs)[::-1][:top_k]
        recommendations = [self.idx_to_item[i] for i in top_indices if probs[i] > 0]
    
        if len(recommendations) < top_k:
            recommendations = self._pad_recommendations(recommendations, last_item, top_k)
        
        return recommendations[:top_k]
    
    def _get_fallback_recommendations(self, last_item: int, top_k: int) -> List[int]:
        
        if self.item_popularity is None:
            return list(range(1, top_k + 1))
        
        top_indices = np.argsort(self.item_popularity)[::-1][:top_k * 2]
        recommendations = [self.idx_to_item[i] for i in top_indices if self.idx_to_item[i] != last_item]
        
        return self._pad_recommendations(recommendations, last_item, top_k)
    
    def _pad_recommendations(self, recommendations: List[int], last_item: int, top_k: int) -> List[int]:
        if len(recommendations) >= top_k:
            return recommendations[:top_k]
        existing = set(recommendations)
        existing.add(last_item)  
        
        if self.item_popularity is not None:
            all_indices = np.argsort(self.item_popularity)[::-1]
            for idx in all_indices:
                item = self.idx_to_item[idx]
                if item not in existing:
                    recommendations.append(item)
                    existing.add(item)
                if len(recommendations) >= top_k:
                    break
        
        if len(recommendations) < top_k and self.item_popularity is not None:
            all_items = list(self.idx_to_item.values())
            np.random.shuffle(all_items)
            for item in all_items:
                if item not in existing:
                    recommendations.append(item)
                    existing.add(item)
                if len(recommendations) >= top_k:
                    break
        
        while len(recommendations) < top_k:
            recommendations.append(-1)  
        
        return recommendations[:top_k]
    
    def get_matrix_stats(self) -> Dict:
        return {
            "size": len(self.item_to_idx),
            "non_zero_count": np.count_nonzero(self.raw_counts),
            "density": np.count_nonzero(self.raw_counts) / len(self.item_to_idx)**2 * 100,
            "mean_prob": self.transition_matrix.mean(),
            "max_prob": self.transition_matrix.max(),
            "min_nonzero_prob": self.transition_matrix[self.transition_matrix > 0].min() if np.any(self.transition_matrix > 0) else 0
        }


class PopularityBaseline:
    def __init__(self):
        self.popular_items = []
        self.item_counts = Counter()
    
    def build_from_sessions(self, train_sessions: List[List[int]]):
        all_items = [item for session in train_sessions for item in session]
        self.item_counts = Counter(all_items)
        self.popular_items = [item for item, _ in self.item_counts.most_common()]
        print(f"\nБейзлайн построен: {len(self.popular_items)} товаров")
        print(f"Топ 10 популярных: {self.popular_items[:10]}")
    
    def get_recommendations(self, last_item: int = None, top_k: int = 10) -> List[int]:
        recommendations = []
        for item in self.popular_items:
            if item != last_item:
                recommendations.append(item)
            if len(recommendations) >= top_k:
                break
        
        while len(recommendations) < top_k:
            recommendations.append(-1)
        
        return recommendations[:top_k]



def hit_at_k(recommendations: List[List[int]], true_items: List[int], k: int = 10) -> float:
    if len(recommendations) != len(true_items):
        raise ValueError("recommendations и true_items должны совпадать по длине")
    
    hits = 0
    for recs, true_item in zip(recommendations, true_items):
        
        if true_item in recs[:k]:
            hits += 1
    
    return hits / len(true_items)

def evaluate_model(model, train_sessions: List[List[int]], test_targets: List[int], model_name: str = "Model"):
    print(f"\nОценка модели: {model_name}")

    
    recommendations = []
    unknown_count = 0
    
    for i, session in enumerate(train_sessions):
        if len(session) == 0:
            last_item = None
        else:
            last_item = session[-1]
        
        recs = model.get_recommendations(last_item, top_k=10)
        recommendations.append(recs)
        if hasattr(model, 'item_to_idx') and last_item not in model.item_to_idx:
            unknown_count += 1
    
    score = hit_at_k(recommendations, test_targets, k=10)
    
    print(f"  Hit@10 = {score:.4f} ({score*100:.2f}%)")
    if unknown_count > 0:
        print(f"  Неизвестных последних товаров: {unknown_count}/{len(train_sessions)} ({unknown_count/len(train_sessions)*100:.1f}%)")
    
    return score, recommendations



def main():
    print("Загрузка данных.")
    sessions = load_data(r"/content/sessions.jsonl")
    unique_items, item_freq = analyze_data(sessions)
    train_sessions, test_targets = train_test_split(sessions)
    print(f"\nРазбиение данных:")
    print(f"  Обучающих сессий: {len(train_sessions)}")
    print(f"  Тестовых целей: {len(test_targets)}")
    transition_model = TransitionMatrixModel(alpha=0.1)
    transition_model.build_from_sessions(train_sessions)
    stats = transition_model.get_matrix_stats()
    print(f"\nСтатистика матрицы переходов:")
    print(f"  Размер: {stats['size']}x{stats['size']}")
    print(f"  Ненулевых ячеек: {stats['non_zero_count']:,}")
    print(f"  Плотность: {stats['density']:.3f}%")
    print(f"  Средняя вероятность: {stats['mean_prob']:.6f}")
    print(f"  Максимальная вероятность: {stats['max_prob']:.4f}")
    baseline = PopularityBaseline()
    baseline.build_from_sessions(train_sessions)
    
    print("\n" )
    print("Оценка качества")
    print("\n")
    transition_score, transition_recs = evaluate_model(
        transition_model, train_sessions, test_targets, "Матрица переходов"
    )
    baseline_score, baseline_recs = evaluate_model(
        baseline, train_sessions, test_targets, "Популярные товары (бейзлайн)"
    )
    
    print("\n" )
    print("Сравнение результатов")
    print("\n")
    print(f"Матрица переходов:  Hit@10 = {transition_score:.4f} ({transition_score*100:.2f}%)")
    print(f"Бейзлайн:           Hit@10 = {baseline_score:.4f} ({baseline_score*100:.2f}%)")
    
    if transition_score > baseline_score:
        improvement = (transition_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else float('inf')
        print(f"\n Модель переходов лучше бейзлайна на {improvement:.1f}%")
    elif transition_score < baseline_score:
        degradation = (baseline_score - transition_score) / baseline_score * 100
        print(f"\n Модель переходов хуже бейзлайна на {degradation:.1f}%")
        print("\n Возможные причины:")
        print("  1. Слишком короткие сессии для обучения переходам")
        print("  2. Сильная популярность нескольких товаров доминирует")
        print("  3. Поведение пользователей близко к случайному")
        print("  4. Нужно увеличить параметр сглаживания alpha")
    else:
        print("\nМодели показывают одинаковый результат")
    
   
  

if __name__ == "__main__":
    main()

