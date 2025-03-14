Analyse de sentiments - prédire la polarité d'un tweet
------------------------------------------------------

L'analyse de sentiments est un problème qui intéresse beaucoup d'entreprises et de personnalités. Il s'agit de découvrir automatiquement, à partir de contenus textuels, quel sentiment dégage un produit, un article de journal, un postage sur un réseau social, etc. Ces informations peuvent être utilisées, par exemple, par une entreprise qui cherche à savoir ce que les clients pensent de son nouveau produit, ou par une personnalité politique qui souhaite connaître l'avis de la population sur un discours ou un projet.

Il existe plusieurs variantes du problème. Dans sa version la plus simple, on peut tout simplement s'intéresser à la **polarité** du texte, c'est-à-dire : est-il positif, négatif ou neutre par rapport au sujet en question ? Il existe aussi des versions plus sophistiquées, où l'on cherche à savoir quel sentiment est dégagé par le texte (haine, tristesse, joie, peur...) ou alors à trouver quels sont les aspects sur lesquels le texte est positif/négatif, et à quel point (par exemple, les clients d'un hôtel apprécient beaucoup la localisation mais critiquent la propreté).

Dans ce projet, nous nous concentrerons sur la version simple, c'est-à-dire, étant donné un texte, il faut prédire s'il est positif (`+`), négatif (`-`) ou neutre (`=`). Cependant, les textes à traiter ne sont pas faciles à analyser, car il s'agit de tweets collectés en ligne à propos du sujet de l'écologie.

Cette tâche a été proposée dans le défit DEFT 2015 (T1). L'[article qui décrit la campagne d'évaluation](http://talnarchives.atala.org/ateliers/2015/DEFT/deft-2015-long-001.pdf) peut être un bon point de départ pour avoir plus d'informations. Les [actes de l'atelier DEFT 2015](http://talnarchives.atala.org/ateliers/2015/DEFT/) contiennent la description des systèmes participants et peuvent vous inspirer pour développer votre prototype.

## Catégories à prédire

Vous devez prédire une catégorie parmi :
* Le signe `+` représente un tweet de polarité **positive** au sujet de l'écologie.
* Le signe `-` représente un tweet de polarité **négative** au sujet de l'écologie.
* Le signe `=` représente un tweet de polarité **neutre**, ni positive ni négative, au sujet de l'écologie.

## Développement à faire

Vous devez écrire un logiciel qui prédit, pour un tweet en français donné en entrée, quelle est sa polarité. Les valeurs possibles sont `+` pour un tweet positif, `-` pour un tweet négatif et `=` pour un tweet neutre. Tous les tweets parlent d'écologie. Par exemple, le tweet ci-dessous doit être classifié comme positif :

`Lutter contre le réchauffement climatique et réduire la facture énergétique`

Alors que celui-ci sera classifié comme négatif :

`c'est dommage d'avoir tellement freiné les énergies alternatives …`

Votre système doit donner en sortie un fichier au même format que l'entrée, avec une copie du tweet suivie d'une tabulation suivie de la catégorie prédite.

## Extensions ou alternatives

D'autres campagnes d'évaluation proposent régulièrement des tâches liées à l'analyse de sentiments, souvent pour la langue anglaise. Nous vous conseillons de regarder les [campagnes SemEval](https://en.wikipedia.org/wiki/SemEval) des dernières années si vous cherchez une inspiration pour une variante, extension ou alternative au sujet type proposé ci-dessus. Par exemple, la détection de parole haineuse est similaire à l'analyse de sentiments, et des datasets comme [hatEval](https://competitions.codalab.org/competitions/19935) sont disponibles gratuitement.
