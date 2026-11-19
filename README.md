# Impacto da Pré-Filtragem de Genes em Bulk RNA-seq

Repositório oficial do trabalho apresentado na **70ª Reunião Anual da Região Brasileira da Sociedade Internacional de Biometria (70ª RBras)**.

**Autor:** William Yuji Kohatsu Tanigava (Universidade Estadual Paulista — PosMAC / FCT)

---

## 🎯 Objetivo

Investigar como a pré-filtragem de genes por variabilidade via *Median Absolute Deviation* ($\text{MAD} \in \{3\text{k}, 5\text{k}, 10\text{k}, 20\text{k}\}$ genes) influencia a preservação topológica, a estabilidade e a identificação de agrupamentos moleculares (PAM50) em dados de *bulk RNA-seq* do **TCGA-BRCA** ($n = 970$ pacientes, $40.761$ genes originais).

---

## 🔬 Metodologia

* **Pré-filtragem por MAD:** Seleção dos top $3.000$, $5.000$, $10.000$ e $20.000$ genes mais variáveis (após remoção de $12\%$ das amostras originais sem anotação PAM50).
* **Redução Linear Prévia:** PCA com $50$ componentes principais.
* **Redução de Dimensionalidade (*Manifold Learning*):** UMAP vs. PaCMAP.
* **Agrupamento Não Supervisionado:** Leiden vs. HDBSCAN (otimizados via *Grid Search*).
* **Métricas de Avaliação:** *Trustworthiness* (topologia), *Silhouette Score* (coesão geométrica), ARI e NMI (concordância biológica com PAM50).
* **Estabilidade:** Avaliação dos modelos selecionados ao longo de $10$ sementes aleatórias.

---

## 📊 Principais Resultados

### Achados-Chave
* **Robustez ao ruído:** Sob alto ruído gênico ($20\text{k}$ genes), o **PaCMAP** supera o **UMAP** em **25%** na separação geométrica (*Silhouette Score*) e mantém **99%** de estabilidade entre sementes.
* **Concordância biológica:** O **PaCMAP** com **$10\text{k}$ genes** atingiu o maior alinhamento com os subtipos intrínsecos **PAM50** ($\text{ARI} = 0,345$; $\text{NMI} = 0,498$).
* **Impacto do *manifold*:** A escolha do algoritmo de redução de dimensionalidade domina a qualidade e estabilidade das partições frente à escolha entre Leiden e HDBSCAN.

### Resumo das Métricas (Média de 10 Sementes)

| MAD | Silhouette (PaCMAP / UMAP) | ARI (PaCMAP / UMAP) | Estab. HDBSCAN (PaC / UMAP) | Estab. Leiden (PaC / UMAP) |
| :---: | :---: | :---: | :---: | :---: |
| **3k** | **0,840** / 0,833 | 0,336 / 0,333 | 0,995 / 0,996 | 0,995 / 0,949 |
| **5k** | 0,831 / 0,823 | 0,336 / 0,325 | 0,997 / 0,997 | 0,997 / 0,997 |
| **10k** | 0,817 / 0,757 | **0,345** / 0,323 | 0,995 / **0,998** | 0,995 / **0,998** |
| **20k** | 0,781 / 0,625 | 0,343 / 0,317 | 0,987 / 0,991 | 0,987 / **0,910** |

> *Nota:* HDBSCAN e Leiden convergem nas métricas de Silhouette e ARI; *Trustworthiness* $> 0,90$ em todos os cenários.

---

## 📁 Organização do Repositório

* `src/`: funções reutilizáveis (pré-filtragem, métricas e utilitários)
* `scripts/`: execução dos pipelines de *Grid Search* e testes de estabilidade
* `notebooks/`: análise exploratória e visualização dos resultados
* `results/figures/`: gráficos finais utilizados no pôster
* `data/`: dados brutos do TCGA-BRCA

---

## 💾 Dados

Os dados brutos estão incluídos neste repositório na pasta ['data/raw`]. Os dados processados e tratados não estão disponibilizados. 

---

## ⚙️ Ambiente e Reprodutibilidade

1. Crie e ative um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt


---

## 📚 Referências

* **PaCMAP:** Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization. *Journal of Machine Learning Research*, 22(201), 1–73.
* **UMAP:** McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.
* **Leiden:** Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9(1), 5233.
* **HDBSCAN:** Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. *PAKDD*, 160–172.
* **PAM50:** Parker, J. S., et al. (2009). Supervised risk predictor of breast cancer based on intrinsic subtypes. *Journal of Clinical Oncology*, 27(8), 1160–1167.