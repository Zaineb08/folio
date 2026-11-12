# Folio

Ask questions of your Zotero library and get answers with page-level citations. By default everything runs on your own PC, and your articles and questions stay on it. You can also choose Claude for answers, which sends the passages found for each question to Anthropic.

## Start it

Double-click **Start Folio.bat**. Your browser opens at <http://127.0.0.1:8765>. Closing the black window stops Folio.

## Install it on a new PC

Folio needs:
- Windows 10 or 11 on an Intel or AMD processor. Windows on ARM laptops, such as Snapdragon ones, aren't supported.
- At least 16 GB of memory.
- About 15 GB of free disk space: 4 GB of libraries, 4.4 GB of search models, and 3.3 or 6.6 GB for the answer model.

A graphics card is optional. NVIDIA cards from the GTX 16 and RTX 20 series onwards speed up search. With an older NVIDIA card, or AMD or Intel graphics, search runs on the processor instead, which is slower but works.

Which answer model to use depends on the PC:

| Your PC | Answer model | What to expect |
|---|---|---|
| An NVIDIA graphics card with 8 GB or more | `qwen3.5:9b` | Fast searches and answers |
| 16 GB of memory, without such a card | `qwen3.5:4b`, or Claude | Searches take several seconds, the first indexing of a library takes a while, and answers from `qwen3.5:4b` come more slowly |
| Less than 16 GB of memory | Not recommended | Folio's search models alone need about 5 GB |

Folio picks the model that suits the PC by itself. If only the other one is installed, it uses that instead. You can always choose another one in its model menu.

1. **Download Folio.** On GitHub, click the green **Code** button, then **Download ZIP**, and unzip it somewhere you'll find again, such as Documents. The repository is private, so you need to be invited to it first.
2. **Double-click Start Folio.bat.** The first start checks for the three programs Folio relies on, and offers to install any that are missing: uv, which sets up Python, Ollama, which runs the answer model, and Zotero. It asks before each one. Then it downloads Python and Folio's libraries, about 4 GB, and opens Folio in your browser.
3. **Put some papers in Zotero.** Open Zotero once so it creates its library, and add papers as usual, or with Folio's **Discover**. To work on the same papers as someone else, share a Zotero group library with them: Folio reads groups too. The first indexing downloads the search models, another 4.4 GB.
4. **Choose the answer model.** Open the model menu at the bottom left and pick the one marked "suits this PC". Folio shows its size, downloads it and switches to it. If the chosen model isn't downloaded yet, a **Download** button appears under the menu. With your own Anthropic API key you can choose Claude instead, and skip Ollama. Search still runs on your PC, and the literature matrix needs a model on your PC.

To install the programs yourself instead, run these, then double-click **Start Folio.bat**:
```bash
winget install --id astral-sh.uv
```
```bash
winget install --id Ollama.Ollama
```
```bash
winget install --id Zotero.Zotero
```

To get a newer version later, download the ZIP again and replace the old folder. Your papers, settings and index are kept elsewhere, so nothing is lost.

## Use it

**Add articles**
- Under **Discover**, search for a topic and click **Add to Zotero**.
- Or add them to Zotero as usual, by dragging in PDFs or using its browser extension.

Either way, Folio notices new articles within about 30 seconds and indexes them by itself, as long as it's running. Anything added while it's closed gets indexed the next time you start it.

**Ask questions.** Pick the kind of answer you want:

| Mode | What you get |
|---|---|
| Answer | A direct answer in a few paragraphs |
| Lit review | A passage organised by theme, for a thesis draft |
| Compare | A table of what each source says, then where they agree and disagree |
| Find gaps | Open questions and limitations the sources name |

The filter button next to the modes limits a question to certain Zotero collections, tags or years. **Ask about this** in the Library limits it to one article.

**Understand a paper.** **Explain this paper**, in the Library or above a PDF page, gives a plain-language guide: why they did it, what they did, what they found, the limitations, terms to know, and questions to ask as you read. It takes under a minute with a graphics card, and longer without one. It's based on 12 passages chosen from across the paper, and says how many that is out of the total, so use it to prepare for reading the paper, not to replace it. Follow-up questions stay on that paper, and you can ask them in French.

**Find a source for a sentence.** Paste a sentence from your draft under **Find a source**. Folio shows the sentence from each paper that comes closest, sorted into what supports it, what may disagree, and what's only related. If nothing fits, it offers to search Discover. It also warns you when a figure in your sentence isn't in the passage, and when a passage is only reporting another paper's work, so you can cite the original.

**Compare papers in a literature matrix.** Under **Literature matrix**, each paper gets a row and each question a column: research question, method, data and participants, main findings and limitations. Click **Fill empty cells** and the model on your PC writes a short note in each cell from that paper's own pages, with page numbers you can click to check. You can keep using Folio while it fills, and stop it at any time without losing what's done.
- **Choose the papers** with the menu at the top, for example only the `phd-core` tag.
- **Add your own columns**, such as "Languages" or "Detection approach", with what to look for in each paper. Only the new cells get filled.
- **Download for Excel** saves the matrix with page numbers instead of citation numbers, and a comment on any cell that needs checking.
- **New papers** get an empty row, and a paper whose PDF changed is marked for filling again.
- **A paper that's in Zotero twice** gets one row, using the copy with the PDF. Folio says how many duplicates it hid. To merge them, open Zotero and click **Duplicate Items** in the left pane.

Filling a whole matrix takes minutes with a graphics card, and many times longer without one or while other heavy programs share it.

**Check the sources.** Click a citation number to see the real PDF page with the passage highlighted. **Open in Zotero** jumps to that page in Zotero.

**Use it in your writing.** **Copy with references** turns citations into author-year form and adds a reference list. **Copy BibTeX** copies entries for the cited articles. Check both against your style guide: they are only as good as the metadata in Zotero, so fix any errors there.

## Choosing a model

Pick the answer model at the bottom left of Folio. Each answer says which model wrote it.

| | Model on this PC | Claude |
|---|---|---|
| **Cost** | Free | Paid per question: 3 to 9 cents in testing, about 5 on average. Each answer shows roughly what it cost. |
| **Privacy** | Nothing leaves your PC | The question and the passages found for it go to Anthropic |
| **Needs** | Nothing more | An Anthropic API key with credit |

To use Claude, create a key at [console.anthropic.com](https://console.anthropic.com/settings/keys), then choose Claude in the model menu and paste it there. A claude.ai subscription doesn't include API access. Folio checks the key with Anthropic and keeps it in Windows Credential Manager, not in a file. Never paste it into a chat.

With Claude selected:
- **Searching stays on your PC.** Only the passages it finds are sent, not your whole library.
- **Claude's safety filters can decline security questions**, even legitimate ones. Folio asks Anthropic to retry on another Claude model when that happens, and the answer says if it did. If every model declines, Folio tells you and discards any partial text.
- **The model on your PC does the side jobs:** rewriting follow-up questions and checking Claude's claims. They stay free, and the check comes from a different model.

## How far to trust the answers

Treat answers as a guide to your sources, not as a source. Read the passage before you cite anything.

Dotted underlines mark claims Folio wants you to verify. After each answer it checks that:
- every cited source exists,
- every number in a claim appears in the cited passage,
- every quotation appears word for word in the cited passage,
- the local model, reading each claim alone beside its passage, agrees the passage supports it.

The first three are exact. The last is a judgment: it can raise false alarms on correct claims, and it has passed a sentence where the model added its own spin. So a flag means "go and look", and a pass doesn't prove a claim is right. When Claude writes, a different model does this check. When the local model writes, it's judging its own work.

**"Your papers only touch on this"** appears when the passages found match a question poorly, and the model is told to report only what they state and not to give advice they don't. In a test of 30 questions, it appeared on 12 of the 14 the papers couldn't answer and on none of the 16 they could. It misses questions that match the topic but need a fact the papers don't contain, such as "which assistant is the most secure?", so no warning doesn't mean the papers settle the question.

**Find a source sorts passages with a model, so it makes mistakes.** By my reading of the results on the local model:
- **Supports:** right for 15 of 17 passages on 10 sentences that played no part in tuning it. It also found 17 of the 18 real supporting passages on the 12 sentences used for tuning. A wrong one is usually about a narrower or different thing than your sentence.
- **May disagree:** unreliable. Right for 13 of 18 on the tuning sentences, but at most 1 of 6 on the new ones. Treat it as a pointer to read, and don't rely on it to find every disagreement.

**The literature matrix is a first draft to check, not a finished table.** By my reading of 55 cells (11 papers) against their pages:
- **42 were right.**
- **7 were partly right**, for example describing only one step of a method, or slipping a limitation into the findings.
- **3 said "Not stated" when the paper does say it.** "Not stated" only means the 4 passages searched didn't say.
- **3 fairly said "Not stated"**, such as participants for a survey.

Cells skip sections about other people's work, so a paper isn't credited with results it only cites. A figure that isn't in the passages gets a dotted underline in Folio and a comment in Excel.

Answers also only reflect what's in your library. If the papers that disagree aren't in it, you'll get a one-sided answer with no warning.

## Known limits

- **Scanned PDFs** have no text layer and can't be searched. They're marked in the Library, and running OCR on the PDF first fixes this. If the item has an abstract, that's still searchable.
- **Some PDFs can't be fetched by Discover.** Publishers often block automatic downloads, even for open-access articles: over a third of attempts failed in testing. Folio still saves the details and abstract, and you can attach the PDF in Zotero yourself. Paywalled articles need your own access.
- **Whole-library questions**, like "what are the main findings?", read a sample of a passage or two per article, not every article in full.
- **Discover needs the internet**, and what you type there goes to OpenAlex, or to Crossref when OpenAlex is busy. Nothing from your library is sent. Claude needs the internet too. Everything else works offline.
- **Figures** aren't read, and tables are read only as plain text.

## How it works

| Step | What happens | Where it runs |
|---|---|---|
| Read Zotero | Works from a private copy of Zotero's database, so Zotero's own files are never changed | CPU |
| Parse PDFs | Handles two columns, running headers and hyphenation, and drops reference lists while keeping any appendices after them, even lettered ones ("A Data", "B Prompts"). It also keeps each passage's position on the page | CPU, in parallel |
| Index | [BGE-M3](https://huggingface.co/BAAI/bge-m3) embeddings, which handle many languages, plus a keyword index | GPU, or CPU without one |
| Search | Meaning and exact terms combined, then reranked by [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | CPU, then GPU if there is one |
| Answer | `qwen3.5:9b` or `qwen3.5:4b` through [Ollama](https://ollama.com), or Claude if you choose it, using only the retrieved passages | GPU if there is one, or Anthropic's servers |
| Check | The citation checks described above | Same as the answer model on your PC |

Without a graphics card, search reranks fewer passages so it stays responsive, and indexing reuses the copy of BGE-M3 that embeds questions, so Folio needs about 5 GB of memory.

To use a different model, install it with Ollama and pick it at the bottom left of Folio.

## Where things are

Everything Folio needs apart from its code is in `%USERPROFILE%\Folio`, for example `C:\Users\you\Folio`:

| Folder or file | What it holds |
|---|---|
| `folio.db` | The search index and your saved conversations. Back this up to keep conversations. The index can be rebuilt. |
| `venv`, `python` | The Python environment |
| `models` | The search models |
| `ollama` | The Ollama program |

The answer model is in `%USERPROFILE%\.ollama`. Your articles stay in Zotero's own folder, usually `%USERPROFILE%\Zotero`.

## For developers

```bash
uv sync
```

```bash
uv run pytest
```

Set `UV_PROJECT_ENVIRONMENT` to `%USERPROFILE%\Folio\venv` first, to use the existing environment. Start with `folio/app.py` for the web API, `folio/search.py` for retrieval and `folio/answer.py` for answering and citation checks.
