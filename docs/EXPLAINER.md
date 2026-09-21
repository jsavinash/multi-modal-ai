# This Project, Explained Simply

*No jargon. If you can follow a recipe, you can follow this.*

This project is a **file-to-answers machine**. You give it your messy real-world files
— PDFs, scanned photos, voice recordings, videos, spreadsheets — and it produces two
useful things:

1. **Searchable knowledge** — so later, a computer can find "the part about boiler
   pressure" without re-reading everything.
2. **Written answers** — you ask a question, it reads the relevant pieces, answers it,
   and saves a neat report you can download.

Picture a **library with four desks**. Every file walks past the desks in order, and
each desk adds something useful.

---

## 1. Why does this exist?

Real information is spread across very different kinds of files, and computers are bad
at reading most of them.

| Your file is… | Without this project | With this project |
|---|---|---|
| A PDF report | You read 200 pages yourself | Split into paragraph-sized pieces, labelled "header / paragraph / footer" |
| A photo of a whiteboard | Nothing searchable | Text is read off the image and stored |
| A 1-hour meeting recording | You listen again | Split into 30-second slices and written down as text |
| A video | You scrub through it | 1 picture per second is grabbed and read |
| A spreadsheet of sensor readings | Just numbers, hard to search | Each row becomes a sentence a computer can search |

The end result: **everything becomes text with meaning attached**, and that text can be
searched, summarised, and reasoned over.

---

## 2. The four desks

### Desk 1 — The Reader (documents)

**Comes in:** PDF, Word (DOCX), plain text (TXT), Markdown (MD)

**Comes out:** the same content, chopped into tidy pieces, each labelled with the page
number and what kind of thing it was (a header, a paragraph, a footer, or "this page was
a scan with no text").

Analogy: a careful assistant retyping a document, keeping the structure intact instead
of dumping one wall of text.

House rules for this desk:

- The file's true type is checked by looking at its **actual bytes**, not the name you
  gave it — so a `.txt` that is secretly a photo gets rejected.
- Maximum file size: **100 MB**.
- Text is cut into pieces of about **512 tokens** (roughly 2,000 characters), with a
  **10 % overlap** so a sentence cut in half still makes sense in both pieces.

### Desk 2 — The Eyes and Ears (images, video, audio)

**Comes in:** JPEG/PNG/WebP/BMP pictures, MP4/AVI/MKV videos, WAV/MP3 audio

**Comes out:** text — words read off images (OCR) and words spoken in recordings
(transcription), each stamped with a time or a page so you know where it came from.

Analogy: a colleague who watches and listens, and writes down what they saw and heard.

House rules (mostly so your computer survives):

- Every picture is shrunk so its longest side is **512 pixels** — small enough to be
  cheap, big enough to still read.
- Videos are sampled at **exactly 1 picture per second** — not every frame.
- Audio is converted to **16 kHz mono** and sliced into **30-second windows**.
- Speech is transcribed by a small model (`whisper tiny`) that is deliberately tiny.
- If a model isn't installed, you don't get an error — you get a note saying
  `ocr_unavailable` or `asr_unavailable` and life goes on.

### Desk 3 — The Librarian (spreadsheets and data files)

**Comes in:** CSV, XLSX, JSON

**Comes out:** the same rows, but (a) rewritten as readable sentences and (b) converted
into a **number-fingerprint** that captures what the row *means*, so similar things can
be found later even if the wording differs.

Analogy: a librarian who refuses to file raw tables and instead writes one index card per
row. A sensor reading row like `timestamp=10:00, sensor=boiler, value=92` becomes the
sentence:

> "At timestamp 10:00, the boiler sensor recorded a telemetry value of 92 units"

Then a number-fingerprint is made from that sentence and filed away.

House rules:

- Numbers in a column are squeezed into a **0-to-1 range** (min–max scaling) so a
  column measured in the thousands doesn't drown out one measured in decimals.
- Latitude/longitude columns are detected automatically and flagged as location data.
- Fingerprints are made in batches of at most **256 rows** at a time, to keep memory low.

### Desk 4 — The Answer Desk (reasoning)

**Comes in:** your question plus a bundle of relevant pieces (found in the library)

**Comes out:** an answer, a list of suggested **actions**, and a **Markdown report file**.

Analogy: a knowledgeable colleague who is only given the most relevant notes — not the
whole library — and who writes up a clean answer.

House rules:

- There is a budget for how much text can be handed over: **8,000 tokens** if the answer
  is generated locally, **32,000** if a cloud service is used.
- If the bundle is too big, the least relevant pieces are **dropped first** — but it
  always keeps at least one piece per type (a document snippet, an image snippet, a data
  snippet), so nothing is silently ignored.
- Absolute maximum: **2 MB** of text. Anything bigger is refused instead of being sent.
- If the cloud is unreachable, it does **not** hang forever. After **3 failures** a
  "fuse" (circuit breaker) trips for **30 seconds**, and the system falls back to a
  simple offline summary, clearly labelled as such.

---

## 3. How a file travels through the system

Speed matters, so the desks don't work in the same room as the front door. Here is the
restaurant version:

1. **You upload a file** at the front door (the web API). It is written to disk in
   1 MB slices — never loaded into memory all at once.
2. **A ticket is printed** and dropped on a **waiting line** for the right desk. This is
   what Celery + Redis are: the waiting line, and the ticket dispenser.
3. **A cook (worker) picks up the ticket** and does the work at their own pace.
4. **The result is written to a file** under `/tmp/multimodal-ingestion/results/`.

Because the line is separate from the front door, your upload gets an instant
"accepted, here is your ticket number" reply instead of waiting for the whole job.

```
 You ──upload──► Front door (web API) ──ticket──► Waiting line (Redis)
                                                     │
                    ┌────────────────────────────────┼────────────────────────────────┐
                    ▼                                ▼                                ▼
             Desk 1: Reader              Desk 2: Eyes & Ears             Desk 3: Librarian
             (documents)                 (pictures/sound/video)          (tables → fingerprints)
                    │                                │                                │
                    └──────────────┬─────────────────┴─────────────────┬──────────────┘
                                   ▼                                   ▼
                       Filing cabinet for                 Desk 4: Answer Desk
                       fingerprints (Qdrant)              (question → answer + report)
```

There are exactly **four waiting lines**, one per desk, so a huge video can never block
a small text file.

---

## 4. The four front doors (APIs)

Each desk has one door you knock on:

| You want to… | Knock here | You immediately get back |
|---|---|---|
| Hand in a document | `POST /api/v1/documents/ingest` | `task_id` + `"status": "queued"` |
| Hand in a picture/video/audio | `POST /api/v1/media/ingest` | `task_id` + `"status": "queued"` |
| Hand in a spreadsheet/data file | `POST /api/v1/tabular/ingest` | `task_id` + `"status": "queued"` |
| Ask a question | `POST /api/v1/reason/query` | `task_id` + `"status": "queued"` |
| Check the service is alive | `GET /health` | `{"status": "ok"}` |

"Queued" means "we have your ticket, we're working on it". Errors are polite and clear:

| What went wrong | What you get |
|---|---|
| File is bigger than 100 MB | `413` — too large |
| File type isn't supported (or lies about its type) | `415` — unsupported |
| Disk is nearly full (under 2 GB free) | `507` — not enough storage |
| The waiting line is down | `503` — try again shortly |

---

## 5. What comes out at the end

Everything is plain files you can open:

| File | What's inside |
|---|---|
| `results/<document-id>.json` | One entry per piece of text, with page numbers and structure labels |
| `results/media_<id>_<name>.json` | The text read from images, plus transcripts of audio with timestamps |
| `results/index_<id>.json` | A summary of a data import: how many rows, where they were filed |
| `results/reasoning_<query-id>.json` | The answer, the suggested actions, how much context was used |
| `results/reports/report_<query-id>.md` | A tidy **Markdown report** you can read or share |
| `results/spool/<id>.json` | A safe "parking lot" copy used when the filing cabinet is offline |

Nothing is invented: every piece carries a `document_id` so you can always trace an
answer back to the file (and page or timestamp) it came from.

---

## 6. The whole point of Desk 3, in one example

You upload this CSV:

```
timestamp,sensor,value
10:00,boiler,92
10:01,boiler,95
```

What happens:

1. **Read** the rows (CSV snapshot).
2. **Rewrite** each row as a sentence: *"At timestamp 10:00, the boiler sensor recorded
   a telemetry value of 92 units"*.
3. **Squeeze** the numbers into 0–1 so mixed units don't distort results
   (92 → 0.0 and 95 → 1.0 for this tiny table).
4. **Fingerprint** each sentence (a list of 384 numbers that captures its meaning).
5. **File** the fingerprints in a named drawer: `tabular_telemetry`.

Now a question like *"when did the boiler get hot?"* can find those rows — even if the
question never uses the word "value".

---

## 7. The house rules that keep your computer alive

A laptop has limits. Instead of crashing, the project works inside a budget.

The clever part: **the rules are not hard-coded to one laptop**. At startup the service
measures how much memory the machine actually has, then adjusts itself. The same
container image therefore works on a small laptop and a big server.

| Desk / service | Bench space it may use | How many cooks at once | Worst case |
|---|---|---|---|
| Documents (Desk 1) | 2 GB | 2 | 2 × 1 GB = 2 GB |
| Eyes & Ears (Desk 2) | 1 GB | 2 | 2 × 384 MB = 768 MB |
| Librarian (Desk 3) | 1 GB | 1 | 768 MB |
| Answer Desk (Desk 4) | 512 MB | 1 | 512 MB |
| Front door (web API) | 512 MB | — | — |
| Filing cabinet (Qdrant) | 1 GB | — | — |
| Ticket machine (Redis) | 256 MB | — | — |
| **Total** | **6.25 GB** | | |

This machine has **8 GB**, so about **1.75 GB** is deliberately left over for macOS and
Docker itself. At startup the service checks every row above and complains loudly if
`cooks × bench space` doesn't fit — so a mistake is caught on day one, not at 3 a.m.

**One warning:** the optional "local brain" mode (`INGEST_LOCAL_LLM_ENABLED=true`) needs
a 16 GB+ machine. Do not switch it on here — it would break the budget above.

---

## 8. What happens when something is missing?

Nothing explodes. The system keeps working and **tells you what it couldn't do**.

| If this is missing | Instead of failing, you get |
|---|---|
| The OCR engine (reads text from pictures) | The picture is still processed; a note says `ocr_unavailable` |
| The speech-to-text model | The audio file is accepted; a note says `asr_unavailable` |
| The "fingerprint" model | A simple built-in fallback is used so the pipeline still runs |
| The filing cabinet (Qdrant) | Fingerprints are parked in `results/spool/` to be filed later |
| Any cloud AI service / API key | A plain offline summary, clearly labelled `fallback_summary` |
| `ffmpeg` (audio converter) | MP3 files report a clear message; WAV files still work |
| All the extra AI libraries | The project still runs — just with fewer abilities |

This "keep going with less" behaviour is deliberate, because a pipeline that dies at
3 a.m. is worse than one that finishes with a clear note.

---

## 9. Do I need a graphics card?

Short answer: **no**. Longer answer, because it's interesting:

- The code can use a **CUDA** GPU (NVIDIA), **MPS** (the Apple M1/M2 graphics chip), or
  plain **CPU**. It picks automatically: CUDA first, then MPS, then CPU.
- Those accelerators need an extra library called **torch**, which is *not* installed by
  default (it is big). Turn it on only if you want it: `pip install torch`
- On *this* 8 GB Apple M1 laptop, plain CPU is genuinely the better choice: measured, the
  graphics chip was **3–4× slower** for this kind of small work (it spends more time
  starting jobs than doing them), and it eats into the same memory budget as everything
  else. So the default setting (`auto`) correctly stays on CPU here.
- Want to see for yourself? `PYTHONPATH=. python scripts/smoke_device.py` prints the
  detection result, a speed comparison, and a correctness check.

So: **CPU is fine. A GPU is optional, and on a small unified-memory laptop it may not
even be faster.** No part of the project depends on it.

---

## 10. How to run it

You need Python and Docker.

```bash
# 1. Set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the ticket machine (Redis)
docker compose up -d redis

# 3. Start the front door
uvicorn app.main:app --reload

# 4. Start a worker for the desk you care about (one terminal each)
celery -A app.celery_app.celery_app worker -l INFO --queues=ingestion --concurrency=2
celery -A app.celery_app.celery_app worker -l INFO --queues=media --concurrency=2
celery -A app.celery_app.celery_app worker -l INFO --queues=index --concurrency=1
celery -A app.celery_app.celery_app worker -l INFO --queues=reasoning --concurrency=1
```

Or start everything at once: `docker compose up --build`.

Then knock on a door:

```bash
curl -F "file=@report.pdf"  localhost:8000/api/v1/documents/ingest
curl -F "file=@photo.jpg"   localhost:8000/api/v1/media/ingest
curl -F "file=@sensors.csv" localhost:8000/api/v1/tabular/ingest
curl -F "file=@context.json" "localhost:8000/api/v1/reason/query?query=What+changed?"
```

Check it's healthy: `curl localhost:8000/health`

Run the checks that prove it works:

```bash
pytest -q                                       # the full test suite
PYTHONPATH=. python scripts/smoke_pdf.py        # a real PDF, end to end
PYTHONPATH=. python scripts/smoke_media.py      # a real PNG + a real WAV
PYTHONPATH=. python scripts/smoke_tabular.py    # a real CSV → sentences → filing
PYTHONPATH=. python scripts/smoke_reasoning.py  # a real question → report
PYTHONPATH=. python scripts/smoke_device.py     # CPU vs graphics chip
```

---

## 11. Jargon, translated

| Fancy word in the code | What it actually means |
|---|---|
| Ingestion | "Taking files in and making sense of them" |
| Modality | "What kind of thing is it — text, picture, sound, or data?" |
| Chunk | "A paragraph-sized piece of a document" |
| Payload | "One packaged result, ready to hand over" |
| Embedding / vector | "A number-fingerprint that captures meaning" |
| Qdrant | "The filing cabinet that stores fingerprints and finds similar ones" |
| Collection | "A labelled drawer in the filing cabinet, one per kind of thing" |
| Canonical document | "The one agreed, tidy shape for a document after it has been read" |
| OCR | "Reading text off a picture" |
| ASR / transcription | "Writing down what was said" |
| Telemetry | "Sensor readings over time" |
| Min–max scaling | "Squeezing numbers so everything sits between 0 and 1" |
| Queue | "The waiting line of tickets" |
| Worker | "The cook who picks up tickets and does the work" |
| Celery / Redis | "The kitchen staffing system / the ticket machine" |
| Task | "One ticket: one file to process" |
| Guardrail | "A house rule that stops the machine hurting itself" |
| Graceful degradation | "Keep going with less when a tool is missing" |
| Circuit breaker | "A fuse that trips after repeated failures, then retests" |
| Context window | "How much text can be handed to the answer desk at once" |
| Pruning | "Dropping the least relevant notes so the rest fits the budget" |
| Container / `mem_limit` | "A box with a fixed amount of bench space" |
| Spool | "A parking lot for results while the filing cabinet is offline" |

---

## 12. Quick answers to common questions

**Is my original file kept?**
No. Uploads are temporary and deleted after processing. Only the extracted text and the
results remain (under `/tmp/multimodal-ingestion/results/`).

**Why did my upload answer "queued" instead of finishing?**
That is by design — the front door accepts instantly, the desks process in the
background. The `task_id` is your receipt.

**A 2-hour video takes a long time. Is that normal?**
Yes. It deliberately reads 1 picture per second and shrinks each one: slow, but memory
stays flat instead of the machine falling over.

**Can it run without internet?**
Yes. Everything except the optional cloud AI service works offline, and questions fall
back to a clearly-labelled offline summary.

**Does it change my files or make surprise internet calls?**
No. Desks 1–3 never call out. Only desk 4 talks outward, and only when you configure an
API key; otherwise it stays local.

**Where do secrets go?**
In a `.env` file, which is never committed. Copy `.env.example` to `.env` and fill it in.

---

## 13. Where to go next

| If you want… | Read this |
|---|---|
| The one-page big picture | [`INDEX.md`](INDEX.md) |
| Documents (Desk 1) | [`PHASE1_SPEC.md`](PHASE1_SPEC.md) · [`PHASE1_DOCUMENTATION.md`](PHASE1_DOCUMENTATION.md) |
| Pictures, sound, video (Desk 2) | [`PHASE2_SPEC.md`](PHASE2_SPEC.md) · [`PHASE2_DOCUMENTATION.md`](PHASE2_DOCUMENTATION.md) |
| Tables and searchable data (Desk 3) | [`PHASE3_SPEC.md`](PHASE3_SPEC.md) · [`PHASE3_DOCUMENTATION.md`](PHASE3_DOCUMENTATION.md) |
| Questions and reports (Desk 4) | [`PHASE4_SPEC.md`](PHASE4_SPEC.md) · [`PHASE4_DOCUMENTATION.md`](PHASE4_DOCUMENTATION.md) |
| Setup and commands | [`../README.md`](../README.md) |

*Every number in this document (100 MB, 512 px, 30 s, 6.25 GB, 8k/32k tokens, and so on)
comes straight from the project's own settings, and the test suite checks them — so this
page and the machine tell the same story.*
