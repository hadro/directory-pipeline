---
marp: true
---

# Directory Pipeline 
**(A personal digital collections project)**


<br>

2026-07-28
Josh Hadro
josh.hadro@gmail.com


---


## What the pipeline does

Takes any item with [a IIIF manifest](https://iiif.io/get-started/how-iiif-works/) and runs through a human-in-the-loop, Large Language Model (LLM)-augmented set of steps with meta-prompting to produce structured data with minimal customization required.

**The primary output:** An HTML/JS table viewer (the Data Explorer) with one row per entry in the source material, including images of the location of the entry on the page and a link back to the originating digital object.

---

![bg fit a screenshot of the digital collections viewer for the volume](tovey.png)

---

![bg fit a screenshot of the data explorer for the 1915 Tovey Brewery Guides](brooklyn-brewery-guide2.png)



---

## Top takeaways — what's new here for digital collections?
- Meta-prompting strategy for custom, item-specific OCR and NER prompts based on sample pages
- Two-pass consensus OCR strategy: old tech for bounding boxes, LLMs for higher-quality OCR, matched via alignment step
    - A critical guard against hallucinations



---



![bg 125% NYPL logo](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmattpoor.com%2Fwp%2Fwp-content%2Fuploads%2F2016%2F09%2Fnew-york-public-library-logo.png&f=1&nofb=1&ipt=7ebe178d598dc51b8c52f53f46cec635223c1713dc1e3a69a6991f4d90254748?text=A)
![bg 40% IIIF logo](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOIP.bvZnLW3MHY8rA8XBUTtABgHaGo%3Fpid%3DApi&f=1&ipt=e3c1c25c1a8eff2fc5f52b58782f7f3f2919cda0ffd87e0bcd0d50dd1add494b?text=B)
![bg fit left:99% LOC logo](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.marefa.org%2Fw%2Fimages%2Fthumb%2Fb%2Fbe%2FLogo_of_the_United_States_Library_of_Congress.svg%2F800px-Logo_of_the_United_States_Library_of_Congress.svg.png&f=1&nofb=1&ipt=e3bb62f2ecc4ed83a0b87890e0c01fccb89ad5ce60d4b6b8f82d06c44cb278f7?text=C)


---

## The Big Picture Challenges for Digital Collections

---

## 1.  Looking at pictures of books is not the same as reading a book. 

JPGs embedded in HTML is better than nothing -- but only a little bit better.

---

![bg](loc_pdf_screenshot2.png)

---

![bg fit a screenshot of the digital collections viewer for the volume](tovey.png)

---

## 2. It's possible ... we in libraries may have digitized *too much.*

We've spent a generation digitizing millions of items, but haven't spent nearly as much energy making them useful to the people who might appreciate them.

---

## Thesis

The tradeoffs have shifted.

For digtized collections:
- Data extraction with computers used to be impossible.
- Then it was doable, but hard and expensive, and not that useful. 
- Then it was doable, and relatively cheap, but useful only in narrow circumstances. 

Now, I think we're on the cusp of doable, cheap, and broadly useful. 

---

## How I think about this

The baseline I'm evaluating against isn't "is this perfect" (OCR is far from perfect!). It's "is this still useful, despite known flaws." For OCR, we answered that question affirmatively decades ago. I think the answer for some collections materials in the library sector, informed by these kinds of enrichments, is getting close to yes.

---

## The Catalysts for this project

- Looking at a Green Book reproduction with my kid 
- Paul Ford's NYTimes OpEd: [The A.I. Disruption We’ve Been Waiting for Has Arrived](https://www.nytimes.com/2026/02/18/opinion/ai-software.html?unlocked_article_code=1.0VA.Pf2C.Y5I6TrcOzxkR&smid=url-share)
- Mark Humphries: [Gemini 3 Solves Handwriting Recognition and it’s a Bitter Lesson](https://generativehistory.substack.com/p/gemini-3-solves-handwriting-recognition)
- [Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents](https://arxiv.org/abs/2504.00414)





---



## Background


---


### Vonnegut's Barber

In Kurt Vonnegut's _Player Piano_ from 1952, there's a minor sub-plot about a barber who sees automation happening all around him, but somehow still not impacting his work as a barber. But he lies awake at night, every night, and spends so much time thinking about how automation could work in terms of barbering if only someone with the right domain knowledge worked on it, that he ends up doing the work to create the barbering robot that kept him up at night.

![bg right:25% contain](https://upload.wikimedia.org/wikipedia/en/6/6f/PlayerPianoFirstEd.jpg)

---

### Navigating the Green Book: 2015-2016

https://beefoo.github.io/greenbook-map/

---

![bg](https://drupal.nypl.org/sites-drupal/default/files/styles/max_width_960/public/blogs/navigating_the_green_books.png)

---




![bg](https://drupal.nypl.org/sites-drupal/default/files/styles/max_width_960/public/blogs/green_book_facsimile.png)




---

## The pipeline

---



### Basic pipeline
&nbsp;&nbsp;&nbsp;&nbsp;Download files 
→ Select sample pages* 
→ Generate OCR + NER prompts 
→ Run LLM OCR 
→ Extract entry data

<br><br><sub>* denotes "human in the loop" step</sub>

---

### Enriched pipeline
&nbsp;&nbsp;&nbsp;&nbsp;Download files 
→ Select sample pages* 
→ Generate OCR + NER prompts 
→ Run LLM OCR 
→ Run layout detection 
→ Match lines 
→ Review alignment*  
→ Extract aligned entry data 


<br><br><sub>* denotes "human in the loop" step</sub>

---

### Then the fun stuff
- Data explorer
- Map interface (after geocoding enrichment)
- Cross-volume comparison
- And more!

---

## Example outputs

- [Date Explorer for 19th Century brewery guides, extracted from a brewery guide](https://hadro.github.io/brewery-guides/explorer#about)


- [The Green Books + Other Travel Guide Explorer (1930-1966)](https://hadro.github.io/green-books/all-volumes)
    - N.B. Lots of design elements layered onto this data explorer



---

# Appendices


---

## Human-in-the-loop interface examples

---

![bg](https://github.com/hadro/directory-pipeline/raw/main/docs/screenshots/web-interface.png)


---

![bg](https://github.com/hadro/directory-pipeline/raw/main/docs/screenshots/select-pages.png)

---

![bg](https://github.com/hadro/directory-pipeline/raw/main/docs/screenshots/review-alignment.png)


---


![bg](https://github.com/hadro/directory-pipeline/raw/main/docs/screenshots/ner-in-cli.png)


---

### Meta-prompting for item-specific extraction guidance

```md
You are a structured data extractor for a digitized historical 
document. Your goal is to identify and extract discrete records 
from the transcribed text of "The National Directory of 
Morticians," a professional registry of funeral homes and 
directors organized by geography.
```
---
```markdown
## Your task
### Entry schema
Each object in the "entries" array must represent a single 
business or practitioner. Inherit the geographic context from the 
headings above the entry. Use the following fields:

- state: The state name (normalized, e.g., "ALABAMA").
- city: The city name.
- county: The county name (e.g., "Henry Co.").
- city_population: The population count listed for that city.
- business_name: The primary name of the funeral home or mortuary.
- personnel: Names of specific directors, managers, or partners 
mentioned (e.g., "Bernie T. Hoff, Mgr.").
```

---

```markdown

## Rules

1. Extract every distinct funeral service provider listed. If a 
boxed advertisement and a text listing refer to the same 
business, merge them into a single entry containing all available 
details.
2. Skip page numbers, running headers, and decorative elements. 
Ignore generic directory filler text (e.g., "Use National 
Directory of Morticians for Accuracy") and "Publishers Notes."
3. Normalize headings: If a heading appears as 
"ARIZONA—Continued", record the state as "ARIZONA".
4. Heading transitions mid-page: When a new City/County heading 
appears (e.g., "BIRMINGHAM—Jefferson Co."), every entry following 
it belongs to that new context. The prior_context only applies to 
entries appearing before the first heading change on the current 
page.
5. If a record spans a page boundary, extract the portion of the 
record present on the current page. If a listing says "(See Ad. 
next page)", include that note in the address or a notes field.
6. Sentinel tokens: If the source text contains [illegible] or 
[blank], copy those tokens verbatim. If a field is simply not 
present, leave it as null or an empty string.

Return only valid JSON. No markdown code fences. No explanatory 
text.
```


---


## Pipeline actors

### What a **human** is doing in this pipeline

- Bringing curiosity to bear
- Exhibiting agency and responsibility
- Employing materials judgement and expertise
- Model selection
- Prompt review
- QA: OCR quality, alignment review, gut check etc.
- Synthesizing, understanding, and getting excited about possibilities

---
## Pipeline actors

### What an **LLM** is doing in this pipeline
- Meta-prompting for prompt-generation (OCR, NER)
- Item-specific OCR extraction
- Handwriting detection 
- Item-specific data extraction

---
## Pipeline actors

### What a **plain old computer** is doing in this pipeline
- File management
- Spread detection (.e.g., microfilm with 2-up pages)
- Column detection
- Layout analysis
- Bounding box identification
- Outlier detection
- LLM OCR to bounding box alignment
- IIIF annotation and content state generation


<!-- ---

## What vibe coding enabled, and what it didn't

Paul Ford, on [vibe coding](https://www.nytimes.com/2026/02/18/opinion/ai-software.html?unlocked_article_code=1.V1A.OOea.4QQ9xxw6vlue&smid=url-share): 

>  [Claude Code] was always a helpful coding assistant, but in November it suddenly got much better, and ever since I’ve been knocking off side projects that had sat in folders for a decade or longer.

For me, the recent spate of tools have meant sustained attention on personal projects that previously languished sometimes for a literal decade.

Put another way: I probably could have always done something sort of like this ... but I was never going to.


---

## Where should this pipeline go next?

### What kinds of things might this unlock? 
 -->

---

## Useful links: 

Code repository: https://github.com/hadro/directory-pipeline 


---
N.B. Gemini and similar API models are currently the best mix of quality and value for OCR/HTR and NER, but nearly all of these things could likely be done locally with tools like Surya/Chandra and models like Qwen 3.5, removing the paid API aspect of this.


---

## Readings

These links substantially inspired my interest in this project: 
- [Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents](https://arxiv.org/abs/2504.00414)
- [Gemini 3 Solves Handwriting Recognition and it’s a Bitter Lesson](https://generativehistory.substack.com/p/gemini-3-solves-handwriting-recognition)
- [The A.I. Disruption We’ve Been Waiting for Has Arrived](https://www.nytimes.com/2026/02/18/opinion/ai-software.html?unlocked_article_code=1.V1A.OOea.4QQ9xxw6vlue&smid=url-share)

---

## AI Disclosure

Claude Code wrote: 
- Most of the code for this project
- Most of the code and git repo documentation for this project 

I wrote:
- Some small amount of the code for this project
- Amendments and edits of the documentation 

I also wrote all of the material for this presentation — no AI was used in these slides (for better or for worse)