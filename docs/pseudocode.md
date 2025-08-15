### Hybrid Recommendation Pseudocode

```
input: filters F, liked books L, limit k

C = apply_filters(Books, F)
if C empty: C = Books

S_content = content_model.score(F, C)
S_collab  = collab_model.score(L, C)

S = alpha * S_content + (1 - alpha) * S_collab
S = apply_diversity_boost(C, S)

R = top_k_by_score(S, k)
return annotate_with_metadata_and_explanations(R, F, sources=[content, collab])
```

### ANN Index Build
```
X = compute_embeddings(Books)
index = AnnIndex(dim=d)
index.build(X)
store(index)
```

### Cold Start
```
if new_user and no L:
  return popular_in_filters(F) or content_based(F)
if new_book and no interactions:
  use content embeddings similarity within genre/theme
```