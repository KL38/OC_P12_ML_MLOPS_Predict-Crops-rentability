# Méthodologie Étape 2 → Étape 3 : du benchmark aux deux endpoints

> Document compagnon de [`plan_P12.md`](plan_P12.md). Le plan principal fixe **quoi** faire et
> **pourquoi** ; celui-ci fixe **dans quel ordre**, entre le premier modèle lancé et l'API qui sert
> deux endpoints. Les renvois `§n` pointent vers le plan principal.

## Contexte

Le notebook `ML CYPD.ipynb` a tout l'outillage en place — `build()`, `run_cv()`, le split groupé et
la config MLflow — mais pas la méthode qui les enchaîne. Ce document répond aussi à la question
laissée ouverte : **un seul modèle, deux endpoints, comment ?**

Deux principes le structurent.

**Le protocole groupé par pays est celui qui décide.** Le split aléatoire mesure surtout la
mémorisation de l'empreinte climatique — (pluie, température) identifie le pays sans ambiguïté
(§13). Sélection, optimisation et évaluation finale se font donc en groupé.

**On évalue le modèle dans les conditions où il sera utilisé.** Toute feature qu'un agriculteur ne
peut pas renseigner n'a pas sa place dans le modèle servi. C'est ce qui a écarté la décomposition
intra/inter des pesticides : `pest_ecart` aurait valu 0 en permanence en production alors qu'il a
une vraie distribution à l'entraînement — un décalage de covariables qui aurait rendu le score
annoncé non représentatif. Le modèle utilise donc `pesticides_kg_per_ha` brut, qui correspond un
pour un à la saisie utilisateur.

---

## Quelle métrique pour quel usage

| Usage | Métrique | Pourquoi |
|---|---|---|
| Sélectionner un modèle (même jeu, même protocole) | **R² groupé** | équivalent au RMSE à jeu de test fixé, et c'est la convention attendue |
| Comparer les trois protocoles entre eux | **RMSE** | seul comparable — voir ci-dessous |
| Rapport métier et interface | **RMSE + MAE en t/ha** | « ±3 t/ha » parle à un agriculteur, « R² = 0,54 » non |
| Diagnostiquer | l'**écart RMSE − MAE** | un RMSE très supérieur au MAE signale que quelques prédictions ratées de loin dominent |

> **Pourquoi le R² ne se compare pas entre protocoles.** R² = 1 − SSE/SST. À jeu de test fixé, SST
> est constant, donc maximiser le R² revient exactement à minimiser le RMSE — les deux ne peuvent
> pas se contredire. Mais le jeu de test groupé et le jeu de test aléatoire n'ont **pas le même
> SST** : une partie de l'écart 0,93 → 0,54 viendrait du dénominateur, pas de la qualité des
> prédictions. Même remarque entre les plis d'un `GroupKFold`, chaque pli contenant des pays
> différents. Le RMSE, erreur absolue en t/ha, n'a pas ce défaut.

---

## Phase A — Benchmark ✅ *(14 runs, faits)*

Tous les runs sur `X_train` / `y_train`. **Le jeu de test n'est pas touché.**

| Protocole | `cv=` | Modèles | Runs |
|---|---|---|---|
| Groupé par pays | `GroupKFold(5)` (défaut) | les 6 | 6 |
| Aléatoire | `KFold(5, shuffle=True, random_state=RANDOM_STATE)` | les 6 | 6 |
| Temporel | liste d'un couple d'indices sur `Year <= 2008` | les 2 finalistes | 2 |

Nommage : `"HistGB — grouped"`, `"HistGB — random"`. `delete_run_if_exists` supprimant par nom, le
protocole **doit** figurer dans le nom du run. Le protocole temporel étant une simple liste
d'indices, `type(cv).__name__` ne dirait que `"list"` : lui passer `tags={"cv": "temporal"}`.

Le groupé sélectionne. L'aléatoire est diagnostique : l'écart entre les deux mesure, modèle par
modèle, le degré de mémorisation — matériau de soutenance directement relié au §13. Le temporel
qualifie le finaliste pour le déploiement (l'app tourne en 2026 sur des données arrêtées en 2013).

**Critère** : `r2_mean` sous `GroupKFold`, `r2_std` en arbitre en cas d'égalité — un modèle stable
d'un pli à l'autre vaut mieux qu'un modèle qui gagne de peu en moyenne.

### Résultats

| Modèle | R² groupé | RMSE groupé | R² aléatoire | RMSE aléatoire | mémorisation |
|---|---|---|---|---|---|
| **HistGB** ✅ | **0,566** | **5,11** | 0,889 | 2,60 | 1,96× |
| CatBoost | 0,533 | 5,29 | 0,918 | 2,24 | 2,36× |
| RandomForest | 0,480 | 5,60 | 0,932 | 2,03 | **2,76×** |
| Ridge | 0,415 | 5,86 | 0,594 | 4,98 | 1,18× |
| Lasso | 0,412 | 5,87 | 0,563 | 5,17 | 1,14× |
| Dummy | −0,164 | 8,39 | −0,160 | 8,42 | **1,00×** |

**Finaliste : HistGB** — meilleur R² groupé *et* le plus stable entre plis.

Trois lectures à retenir, toutes reprises dans le graphe récap (`figures/recap_benchmark.png`) :

1. **Le classement s'inverse entre protocoles.** RandomForest est premier en aléatoire et troisième
   en groupé ; HistGB fait le trajet inverse. Sélectionner en CV aléatoire aurait retenu le modèle
   qui généralise **le plus mal** aux pays inconnus. C'est la démonstration la plus directe du §13.
2. **Le Dummy valide la comparaison.** 8,393 en groupé contre 8,418 en aléatoire, soit 0,3 % d'écart :
   les deux jeux de test ont la même variance, donc la chute 0,93 → 0,48 est réelle et ne vient pas
   du dénominateur du R². Sans ce témoin, la comparaison inter-protocoles ne serait pas défendable.
3. **Le gradient de mémorisation est propre** : 1,00× pour le Dummy qui ne peut rien mémoriser,
   ~1,15× pour les linéaires, 2 à 2,8× pour les arbres.

Le protocole temporel, lui, ne se lit **pas** contre le groupé : il ne groupe pas par pays, donc ses
pays de test sont connus du modèle. Il se compare à l'aléatoire, et l'écart isole le coût du temps
seul — RMSE 2,60 → 3,48, soit **+34 % pour 5 ans d'horizon**, contre +96 % pour un pays inconnu. La
difficulté du problème est spatiale, pas temporelle. Ce chiffre sert au §5 du plan principal.

`run_cv` logue au passage un **R² par pli** en artefact (`r2_par_pli.png`). `r2_std` dit à quel
point le modèle est instable, ce graphique dit **où** : un pli qui décroche isolément signale un
groupe de pays sur lequel le modèle échoue, ce qu'une moyenne masque.

### Le récapitulatif du benchmark — une figure, à la fin

Une fois les 10 runs faits, **un seul graphique** : R² par modèle, deux barres chacun (groupé et
aléatoire), écart-type en barre d'erreur.

Il raconte toute l'histoire en une image — quel modèle gagne, et surtout **quel modèle mémorise le
plus**, l'écart entre ses deux barres. C'est la figure qui part au rapport métier et à la slide de
soutenance.

> Pas de graphique de diagnostic par run : onze figures, dont celles du Dummy et du Ridge, seraient
> du bruit. Et `cross_validate` ne renvoie pas les prédictions — tout graphique de résidus impose un
> `cross_val_predict` supplémentaire. Réservé au finaliste, en Phase D.

---

## Phase B — Ablation pluviométrie · ~~1 run~~ **écartée**

Prévue, puis retirée. La raison est dans la formulation initiale de cette phase : *« l'ablation sert
à chiffrer ce qu'elle apporte, pas à décider de la garder »*. Autrement dit, **aucun résultat
possible ne changeait de décision** — la pluviométrie reste dans le modèle servi quoi qu'il arrive,
c'est une saisie utilisateur, et sans elle `/recommend` ne répond plus qu'à la température.

Et la question qu'elle posait est déjà couverte : la **permutation importance de la Phase E** donne
la contribution de toutes les variables d'un coup, celle-ci comprise.

> *Nuance gardée pour la soutenance* : ablation et permutation ne mesurent pas la même chose. La
> première réentraîne **sans** la variable, la seconde la mélange dans un modèle **déjà entraîné**.
> Les deux divergent quand les variables sont corrélées — et ici la température encode aussi le
> pays, donc un modèle réentraîné sans la pluie peut compenser. Si la question tombe, c'est un
> `run_cv` de cinq secondes.

---

## Phase C — Optimisation (1 run)

`RandomizedSearchCV` sur le finaliste, **sous `GroupKFold(5)`, jamais en CV aléatoire**. Une
recherche en CV aléatoire sélectionnerait la configuration qui mémorise le mieux les empreintes de
pays — exactement ce qu'on cherche à éviter.

Priorité à la régularisation : 14 371 lignes mais 109 profils climatiques effectifs, le risque est
le sur-apprentissage du pays. Profondeur / nombre de feuilles, minimum d'observations par feuille,
régularisation L2, taux d'apprentissage.

Passer `groups=groups_train` à `.fit()`.

---

## Phase D — Évaluation finale (une seule fois)

Modèle optimisé réentraîné sur tout `X_train`, puis évalué **une seule fois** sur `X_test` /
`y_test` — les 22 pays jamais vus.

Publier le **RMSE et le MAE en t/ha** sous les trois protocoles, plus le R² groupé comme indicateur
de variance expliquée. C'est le RMSE qui porte la comparaison entre protocoles.

> `X_test` ne sert à arbitrer aucun choix. Un seul passage, à la fin. S'il déçoit, on le rapporte
> tel quel — on ne retourne pas ajuster.

### Métriques par culture

Le RMSE global est **vraisemblablement** dominé par les tubercules : à erreur relative comparable,
une erreur de 20 % vaut 3,2 t/ha sur la pomme de terre (16 t/ha médians) contre 0,25 sur le sorgho
(1,26), soit un facteur ~160 une fois au carré. Vraisemblablement, pas certainement — c'est une
hypothèse d'échelle, et **ces métriques servent à la vérifier plutôt qu'à la supposer**.

> ⚠️ **En relatif, pas en t/ha.** Un RMSE par culture exprimé en tonnes reproduirait le biais qu'on
> cherche à corriger. Soit 4,0 t/ha d'erreur sur la pomme de terre et 0,6 sur le sorgho : **lu en
> tonnes, le sorgho semble six fois mieux prédit**. Mais rapportées au niveau de chaque culture —
> 16 t/ha médians contre 1,26 — ces mêmes erreurs valent **25 % et 48 %** : c'est en réalité le
> sorgho le plus mal prédit. Le classement s'inverse selon l'unité choisie.

**Métrique retenue : la RMSLE** (*root mean squared logarithmic error*), sur `ln(vrai) − ln(prédit)`.
Une différence de logs étant un rapport, l'erreur est **relative par construction** — et c'est
exactement la quantité que le modèle minimise, puisqu'il s'entraîne sur `log(y)`.

Elle est aussi **symétrique**, ce que le pourcentage n'est pas : surestimer d'un facteur 2 et
sous-estimer d'un facteur 2 donnent la même amplitude (±0,693), là où le pourcentage donnerait
100 % et 50 %. Le pourcentage plafonne mécaniquement à 100 % en sous-estimation alors qu'il est
illimité en surestimation — il pénalise donc plus lourdement les modèles audacieux.

**Pour la rendre lisible** : `exp(RMSLE)` donne un **facteur** d'erreur typique. Une RMSLE de 0,22
donne `exp(0,22) = 1,25` — la prédiction tombe typiquement entre `vrai/1,25` et `vrai × 1,25`. C'est
sous cette forme exponentiée que le chiffre va au rapport métier et à l'interface, le log restant
interne.

> ⚠️ **Un facteur, jamais un pourcentage.** Traduire `exp(RMSLE) − 1` en « ±25 % » est faux dès que
> l'erreur grandit : l'écart vers le bas vaut `1 − exp(−RMSLE)`, qui n'est pas le même nombre. Une
> RMSLE de 1,19 donne **+229 % vers le haut mais −70 % vers le bas** ; annoncer « ±229 % »
> prétendrait à une erreur impossible, un pourcentage plafonnant à −100 % en sous-estimation.
> C'est exactement l'asymétrie que le paragraphe ci-dessus reproche au pourcentage — **elle ne
> disparaît pas en exponentiant**, il faut garder le facteur.

> 🔴 **Ne pas utiliser `sklearn.metrics.root_mean_squared_log_error`.** La convention standard de la
> RMSLE calcule sur `log(1 + y)`, pour tolérer les zéros — et ce `+1` casse la métrique sur nos
> données, qui descendent à 0,005 t/ha. Vérifié :
>
> | Cas | sur `ln(y)` | sklearn (`log1p`) |
> |---|---|---|
> | Pomme de terre + sorgho, tous deux à +25 % | 0,2231 → facteur **1,250** ✓ | 0,1754 → facteur 1,192 ✗ |
> | Facteur 2 sur une petite valeur (0,05 vs 0,10) | 0,6931 = ln(2) ✓ | **0,0465 ≈ zéro** ✗ |
>
> Le second cas est éliminatoire : une erreur d'un facteur 2 sur une culture à faible rendement
> serait rapportée comme nulle, alors que c'est précisément ce que la métrique par culture doit
> révéler. Le `+1` écrase les petites valeurs et redonne un comportement absolu. **Calcul manuel sur
> `np.log`** — la cible étant strictement positive (min 0,005), le `+1` n'a aucune utilité ici.

Afficher **l'effectif de test à côté de chaque chiffre**. La couverture est très inégale : 478
pommes de terre et 488 maïs, mais seulement 115 ignames et **46 plantains**. Une métrique sur 46
observations est bruitée : ne rien conclure dessus, sous peine de remplacer une réserve honnête par
un chiffre faussement précis.

Ce que ça apporte :

1. **Ça chiffre la réserve que le §7 pose aujourd'hui à la main** (« afficher une réserve sur les
   cultures peu documentées »). La phrase devient un nombre affichable à côté de chaque
   recommandation.
2. **Ça vérifie que le passage au log a fait son travail.** Il a été choisi pour que la perte ne
   soit pas dominée par les tubercules : si les erreurs relatives sont du même ordre d'une culture
   à l'autre, c'est réussi. Sinon le choix est à rediscuter — et aucun chiffre global ne le
   révélerait.
3. **Ça qualifie `/recommend`.** L'endpoint classe dix cultures ; si le modèle est fiable sur cinq
   et hasardeux sur les autres, la moitié basse du classement est trompeuse.

C'est une **caractérisation du modèle retenu, pas un critère de sélection** : à ne pas remonter dans
le benchmark de la Phase A, où dix métriques sur onze runs rendraient MLflow illisible.

### Prédit vs réel, coloré par culture — le graphique de diagnostic

Sur le finaliste uniquement, à partir de prédictions **hors-pli** (`cross_val_predict`), avec la
diagonale `y = x` en repère.

C'est le diagnostic le plus riche en régression, et ici il répond à une question précise : **le
modèle est-il biaisé différemment selon la culture ?** Si le nuage des pommes de terre est
systématiquement sous la diagonale et celui du blé au-dessus, on voit immédiatement pourquoi le
classement de `/recommend` peut s'inverser — avant même de calculer le regret ci-dessous.

Il montre aussi si le passage au log a fait son travail : une dispersion visuellement comparable
d'une culture à l'autre, c'est réussi.

### Regret économique — la métrique qui évalue `/recommend`

Le RMSE évalue la **régression**. Il ne dit rien de ce que fait réellement l'endpoint, qui ne produit
pas un rendement mais un **ordre**. Or des erreurs de rendement modestes suffisent à inverser deux
cultures : un modèle peut avoir un excellent RMSE et recommander la mauvaise culture.

Exemple, sur un contexte où blé et pomme de terre sont tous deux cultivés :

```
                    rendement REEL    rendement PREDIT    prix
Pomme de terre          8 t/ha             5 t/ha        200 EUR/t
Ble                     4 t/ha             5 t/ha        250 EUR/t

classement reel   : pomme de terre 1600, ble 1000  -> la pomme de terre gagne
classement predit : pomme de terre 1000, ble 1250  -> le modele recommande le ble
regret            : 1600 - 1000 = 600 EUR/ha
```

Les prix ne sont pas la source de l'erreur — ils sont identiques des deux côtés. C'est l'erreur de
rendement qui inverse l'ordre.

**Un contexte = un couple (pays, année).** C'est exact ici : à pays et année fixés, la pluie, la
température et les pesticides sont tous constants. Seul `Item` varie. Un groupe (pays, année) du jeu
de test est donc littéralement « le même contexte climatique, plusieurs cultures ».

**Algorithme**, sur `X_test` uniquement :

1. Prédire sur tout `X_test`.
2. Grouper par (pays, année) — la clé se reconstruit avec `groups_test` et `X_test["Year"]`.
3. Écarter les groupes à une seule culture : on ne classe pas un singleton.
4. Dans chaque groupe : `profit = prix[culture] × rendement − coût[culture]`, calculé **deux fois**,
   une fois sur le rendement réel et une fois sur le rendement prédit.
5. Comparer `argmax(profit prédit)` — ce que le modèle recommande — à `argmax(profit réel)`.
6. Agréger : **% de top-1 correct** et **regret** = `profit réel de l'optimum − profit réel du choix
   recommandé`, en €/ha.

**Le tableau à produire :**

| Stratégie | top-1 correct | regret médian (€/ha) |
|---|---|---|
| Choix aléatoire parmi les cultures du contexte | ~1/k | — |
| **Classement statique** (toujours la culture la plus rentable en médiane globale) | — | — |
| **Le modèle** | — | — |

> Le classement statique est **le baseline qui compte**. Le §3 a rejeté cette option précisément
> parce qu'un palmarès figé ne réagit pas au contexte — le regret est ce qui le démontre chiffres en
> main. **Si le modèle ne bat pas le classement statique, `/recommend` n'a aucune valeur ajoutée.**
> C'est le test le plus dur du projet, et le plus honnête.

**Limites à énoncer avec le résultat :**

- Les **prix sont une hypothèse** — aucune source n'en contient. Le regret absolu en €/ha est donc
  indicatif ; c'est la **comparaison entre stratégies** qui porte l'information, les prix étant
  identiques pour les trois lignes du tableau.
- On ne classe que **parmi les cultures observées à ce contexte**, jamais les dix : on ne connaît
  pas le rendement réel d'une culture qui n'a pas été cultivée là-bas.
- Avec ~2 821 lignes de test réparties sur 22 pays × 23 années, la plupart des contextes ont
  plusieurs cultures — le nombre de contextes exploitables sera confortable, mais à afficher.

*Si le temps le permet : refaire tourner avec des prix ±30 % pour vérifier que la conclusion « le
modèle bat le statique » ne tient pas à la table de prix inventée.*

---

## Phase E — Interprétation

**Permutation importance** sur le test groupé. On s'attend à voir `Item` dominer largement — c'est le
η² de 53 % du §13, retrouvé du côté du modèle.

Les preuves du §13 sur le caractère non causal des pesticides (corrélation partielle, test de signe,
différences premières) restent **portées par le notebook d'EDA**, pas par le modèle : la
décomposition ayant été écartée, le modèle ne peut plus les démontrer lui-même. Ce n'est pas une
perte, c'est une séparation des rôles — l'analyse démontre, le modèle sert.

### Annotation manuelle des runs MLflow

Le brief l'exige à deux endroits — *« expérimentations annotées dans MLflow »* et *« annoter les
résultats significatifs »*. C'est un point de contrôle qualité explicite, facile à oublier.

Dans l'interface MLflow, chaque run a un champ **Description** éditable (il alimente le tag
`mlflow.note.content`). À remplir **à la main, après coup** — on ne peut pas savoir à l'avance quel
résultat sera significatif. Trois runs méritent une note :

| Run à annoter | Ce qu'on y écrit |
|---|---|
| Le modèle retenu | pourquoi lui plutôt que les autres, et son écart groupé/aléatoire |
| Le plus grand écart groupé/aléatoire | ce qu'il révèle sur la mémorisation de l'empreinte climatique |
| Le run temporel du finaliste | ce que coûte un horizon de 5 ans (+34 % de RMSE), et ce que ça dit du décalage 2013 → 2026 (§5 du plan principal) |

Les captures de ces runs annotés partent directement au rapport métier — c'est le livrable
« screenshots MLflow » du brief.

> **Hors périmètre : les intervalles de prédiction.** Le brief ne demande nulle part de
> quantification d'incertitude. Si la question surgit en soutenance, la piste est la prédiction
> conforme (bibliothèque MAPIE) avec calibration **groupée par pays** — une calibration aléatoire
> donnerait des intervalles trop étroits, le modèle connaissant déjà ces pays.

---

## Phase F — Le lien avec les deux endpoints

**C'est un seul modèle, appelé deux fois différemment.**

```
model_B.joblib   (pipeline complet : encodage + cible en log)
      |
      +-- POST /predict    : 1 ligne  (culture choisie + contexte) -> 1 rendement
      |
      +-- POST /recommend  : 10 lignes (une par culture, MEME contexte) -> 10 rendements
                             -> filtre domaine de validite (p5/p95)
                             -> profit = prix x rendement - cout   [cote Streamlit]
                             -> tri decroissant
```

`/recommend` n'est rien d'autre qu'un `/predict` appelé sur 10 lignes d'un coup, suivi d'une couche
métier. Aucun second modèle, aucun second entraînement.

### Ce que l'API construit à partir des saisies

| Saisie utilisateur | Feature du modèle |
|---|---|
| pluviométrie de la région (mm/an) | `average_rain_fall_mm_per_year` |
| température moyenne (°C) | `avg_temp` |
| intensité de pesticides (kg/ha) | `pesticides_kg_per_ha` |
| culture (pour `/predict`) | `Item` |
| — | `Year` = **année courante**, ajoutée côté serveur |

Correspondance directe, une saisie par feature — c'est le bénéfice d'avoir renoncé à la
décomposition.

**`Year` n'est pas codée en dur à 2013.** HistGB est un arbre : il plafonne sur son dernier seuil, appris
sur des données arrêtées en 2013. La prédiction pour 2026 est donc **identique, au bit près**, à celle de
2013 — figer la date ne changerait aucun résultat, alors qu'une constante `2013` dans le code
suggérerait un traitement qui n'existe pas.

Le décalage temporel est porté par un **disclaimer permanent dans l'application** (§7 du plan principal),
et par aucune correction de tendance. L'hypothèse sous-jacente — famille de modèles à base d'arbres — et
le chiffrage du biais (~+19 % de progrès agronomique manqué sur 2013-2026) sont détaillés à la décision
n°2 du §5.

### Artefacts produits à l'Étape 2, consommés à l'Étape 3

| Artefact | Rôle |
|---|---|
| `models/model_B.joblib` | le pipeline entraîné |
| `models/domaine_validite.json` | p5/p95 de `avg_temp` et de la pluie **par culture**, calculés sur le train |
| `src/preprocessing.py` | `build_features(...)` — construit la ligne attendue par le modèle |

`build_features` doit être **partagée entre le notebook et l'API**, sinon les deux divergeront.

> ⚠️ `Item` est une colonne `category`. Construite à la volée côté API depuis une seule valeur, elle
> n'aurait pas les mêmes catégories qu'à l'entraînement et l'encodage serait décalé.
> `build_features` doit fixer explicitement la liste des 10 cultures. C'est aussi ce que la
> `signature` MLflow protège — d'où l'importance de la passer au `log_model`.

### Le filtre de domaine de validité

Un modèle à base d'arbres ne refuse jamais de prédire. À 10 °C et 700 mm, il renverra un rendement de
manioc plausible à l'écran alors qu'aucune observation n'existe dans cette fenêtre (7 cultures sur 10
seulement y sont documentées).

`/recommend` ne classe donc que les cultures dont le contexte demandé tombe dans les percentiles
p5–p95 observés. Les autres sont écartées avec une mention explicite, jamais silencieusement.

---

## Vérification

1. **Phase A** ✅ — les 14 runs visibles dans `crop_yield_B`, triables par `r2_mean`, filtrables par
   le tag `cv`. Le Dummy doit avoir un R² **négatif** en groupé : c'est le cas (−0,164). S'il était
   positif, le split serait suspect.
2. **Phase C** — le R² groupé du modèle optimisé ≥ celui du même modèle par défaut. Sinon la
   recherche a tourné sur le mauvais `cv`.
3. **Phase D** — `X_test` n'apparaît qu'une fois dans tout le notebook, et après la Phase C.
4. **Phase F** — deux tests d'intégration :
   - `/recommend` sur un contexte tempéré (10 °C, 700 mm) ne renvoie **ni** manioc, **ni** igname,
     **ni** plantain ;
   - `/predict` sur une culture donnée renvoie **exactement** la valeur que `/recommend` affiche pour
     cette culture au même contexte. C'est ce qui prouve que les deux endpoints partagent bien le
     même modèle.

---

## Hors périmètre

Rapport métier, support de soutenance, CI/CD et Docker restent aux §6, §8 et §10 du plan principal.
La table prix/coûts reste côté Streamlit (décision n°2) : elle sert au regret de la Phase D et à
l'affichage, jamais à l'entraînement.
