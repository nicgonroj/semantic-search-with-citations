# semantic-search-with-citations

Búsqueda semántica sobre documentos legales y corporativos que **responde en español citando el documento y
la sección de origen**, **se abstiene cuando la evidencia es débil** y **rechaza cualquier respuesta que cite
fuentes que no leyó**. Es una recreación, con datos 100 % sintéticos, de un asistente que diseñé y llevé a
producción sobre cientos de documentos legales: el tiempo medio de una consulta del equipo legal bajó de horas
a minutos, y la última palabra la siguió teniendo el equipo legal.

> Este repositorio no contiene código, datos ni nombres de ningún cliente. La empresa («Compañía Ejemplo
> S.A.»), sus 40 políticas, contratos y procedimientos y las preguntas de evaluación son inventados.

## El problema

Un equipo legal responde todos los días consultas internas del tipo «¿con cuánto aviso podemos terminar este
contrato?» o «¿quién firma un compromiso de 3.000 UF?». La respuesta está en algún documento entre cientos de
políticas, contratos y procedimientos, y encontrarla toma horas. Un buscador por palabras devuelve documentos,
no respuestas. Un chatbot genérico responde rápido, pero **puede inventar una cláusula o una fuente**, y en un
área legal eso es peor que no responder.

## La solución

```text
pregunta ──► recuperador (BM25 o híbrido) ──► ¿hay evidencia? ──no──► abstención: «no sé, deriva a Legal»
                 top-k secciones                  │ sí
                                                  ▼
                                     generador (extractivo o LLM)
                                                  │
                                                  ▼
                                     verificación de citas ──inválidas──► revisión humana
                                                  │ válidas                (respuesta rechazada)
                                                  ▼
                               respuesta citada [DOC §N] · validación legal: pendiente
```

- **Trozos citables** (`corpus.py`): cada documento se divide por sección, no por cantidad de caracteres.
  Así cada pasaje tiene una cita estable, como la usaría un abogado: `[POL-005 §2]`.
- **Recuperación sin dependencias** (`retriever.py`): BM25 implementado con biblioteca estándar, con
  normalización y stemming liviano para español (`text.py`). Con `OPENAI_API_KEY` se puede usar un
  recuperador híbrido que fusiona BM25 y embeddings por Reciprocal Rank Fusion.
- **Umbral de «no sé»** (`pipeline.py`): si los términos importantes de la pregunta (ponderados por IDF) no
  aparecen en ningún pasaje recuperado, el sistema se abstiene sin llamar al generador.
- **No inventar fuentes** (`citations.py`): una respuesta solo se acepta si trae citas, si cada una
  corresponde a un pasaje recuperado para esa pregunta y si cada párrafo tiene la suya. Una cita a un
  documento que existe pero que no se recuperó también se rechaza. La respuesta rechazada queda guardada para
  auditoría y la consulta pasa a una persona.
- **Generadores intercambiables** (`answerers.py`): uno extractivo, sin red ni claves, que devuelve oraciones
  textuales de los pasajes con su cita; OpenAI o Anthropic (vía `urllib`, sin SDK) con citas obligatorias.
  Los tres pasan por la misma verificación.
- **Evaluación automatizada** (`evaluate.py`): recall@k, MRR, respuestas citadas correctamente y abstención
  correcta en preguntas que el corpus no responde.

## Resultados del modo sin red

Corpus de 40 documentos ficticios (18 políticas, 12 contratos y 10 procedimientos; 134 secciones) y 42
preguntas de evaluación (`data/eval_questions.jsonl`): 32 con una sección correcta esperada y 10 que el corpus
no responde, algunas a propósito parecidas a temas que sí existen. BM25 + generador extractivo, k = 5 y umbral
de cobertura 0,35.

| Métrica | BM25 + extractivo |
|---|---:|
| Recall@1 / @3 / @5 | 90,6 % / 93,8 % / 96,9 % |
| MRR | 0,930 |
| Respuestas citadas correctamente (32 con respuesta) | 90,6 % |
| Respondidas con una cita incorrecta | 1 |
| Abstenciones en preguntas que sí tenían respuesta | 2 |
| Abstención correcta en preguntas sin respuesta (10) | 80 % |
| Respuestas rechazadas por citas inválidas | 0 |

El umbral es una decisión de negocio, no un detalle técnico. `python -m semsearch sweep` muestra el costo de
cada valor sobre el mismo set:

| Umbral | Citadas correctamente | Cita incorrecta | Abstenciones indebidas | Abstención correcta |
|---:|---:|---:|---:|---:|
| 0,20 | 93,8 % | 1 | 1 | 40 % |
| 0,35 | 90,6 % | 1 | 2 | 80 % |
| 0,45 | 71,9 % | 1 | 8 | 90 % |
| 0,50 | 71,9 % | 1 | 8 | 100 % |

Los errores muestran dónde está el límite de un recuperador léxico:

> «Un periodista me llamó para pedirme una entrevista, ¿qué hago?» → abstención. La política habla de
> «medios de comunicación» y «prensa»; ninguna palabra de la pregunta aparece en el corpus.

> «¿Qué garantía tienen los productos vendidos a consumidores finales?» → respondió con una cláusula del
> contrato de distribución que comparte palabras, pero no responde. Es el caso que la verificación de citas
> no puede detectar: la cita existe y fue recuperada, pero no sostiene lo que se pregunta.

El primero lo resuelven los embeddings (`--retriever hybrid`); el segundo, un LLM que pueda decir
`NO_ENCONTRADO` y, siempre, la validación final del equipo legal. No publico cifras de los modos con LLM o
embeddings que no se puedan reproducir aquí sin claves; corre el mismo set para comparar.

## Cómo usarlo

```bash
PYTHONPATH=src python -m semsearch ask "¿Puedo volar en clase ejecutiva?"
PYTHONPATH=src python -m semsearch eval            # métricas del set de evaluación
PYTHONPATH=src python -m semsearch sweep           # barrido del umbral de abstención

# con un LLM o embeddings (opcional)
export ANTHROPIC_API_KEY=...       # modelo por defecto: ANTHROPIC_MODEL=claude-sonnet-5-5
PYTHONPATH=src python -m semsearch eval --answerer anthropic
export OPENAI_API_KEY=...          # OPENAI_MODEL y OPENAI_EMBEDDING_MODEL son configurables
PYTHONPATH=src python -m semsearch eval --retriever hybrid --answerer openai

python -m unittest discover -s tests
```

Cada respuesta indica su estado (`respondida`, `abstencion` o `revision_humana`), las citas, los pasajes
recuperados con su puntaje, la cobertura y el motivo. La evaluación queda en `output/eval_results.jsonl`.

## Qué llevé a producción en el caso real

- Ingesta de cientos de documentos legales y corporativos, troceados por sección e indexados en Azure
  Cognitive Search.
- Respuestas en español generadas con Azure OpenAI GPT-4, siempre citando el documento fuente.
- Backend en Python/FastAPI y una interfaz en React para el equipo legal.
- El asistente propone y cita; el control final de cada respuesta lo mantiene el equipo legal. El tiempo medio
  de consulta bajó de horas a minutos.

## Stack

Python 3.10+ (solo biblioteca estándar) · BM25 · OpenAI / Anthropic (opcional) · unittest

---

**Nicolás González Rojas** · Head of Data & BI · [LinkedIn](https://www.linkedin.com/in/nicolas-gonzalez-rojas)
