import pytest
from peptide_extractor.providers import NCBIProvider

SAMPLE_PUBMED_XML = """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">31000001</PMID>
      <Article PubModel="Print-Electronic">
        <ArticleTitle>Self-Assembly of <i>Tetrapeptides</i>: A Structural Analysis</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Short peptides are capable of forming diverse nanostructures.</AbstractText>
          <AbstractText Label="METHODS">We evaluated assembly kinetics via spectroscopy.</AbstractText>
          <AbstractText Label="RESULTS">The tetrapeptide formed stable amyloid-like fibrils.</AbstractText>
          <AbstractText Label="CONCLUSIONS">Valuable for biomaterials applications.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">31000001</ArticleId>
        <ArticleId IdType="doi">10.1000/182</ArticleId>
        <ArticleId IdType="pmc">pmc-id: 70001;</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">31000002</PMID>
      <Article PubModel="Print-Electronic">
        <ArticleTitle>Hydrogel Formation in Diphenylalanine Derivatives</ArticleTitle>
        <Abstract>
          <AbstractText>Diphenylalanine motifs readily form hydrogels under physiological conditions.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">31000002</ArticleId>
        <ArticleId IdType="doi">10.1000/183</ArticleId>
        <ArticleId IdType="pmc">PMC70002</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

SAMPLE_PMC_XML = """<?xml version="1.0" ?>
<pmc-articleset>
  <article>
    <front>
      <article-meta>
        <article-id pub-id-type="pmid">32000003</article-id>
        <article-id pub-id-type="doi">10.1000/184</article-id>
        <article-id pub-id-type="pmcid">PMC70003</article-id>
        <title-group>
          <article-title>Designed self-assembling peptide hydrogels</article-title>
        </title-group>
        <abstract>
          <p>This is a paragraph-based abstract from PMC format.</p>
        </abstract>
      </article-meta>
    </front>
  </article>
</pmc-articleset>
"""

SAMPLE_MINIMAL_ARTICLE_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE">
      <PMID>33000004</PMID>
      <Article>
        <ArticleTitle>Minimal Article Without Abstract</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


@pytest.fixture
def sample_pubmed_xml():
    return SAMPLE_PUBMED_XML


@pytest.fixture
def sample_pmc_xml():
    return SAMPLE_PMC_XML


@pytest.fixture
def sample_minimal_article_xml():
    return SAMPLE_MINIMAL_ARTICLE_XML


@pytest.fixture
def ncbi_provider():
    return NCBIProvider(api_key="test_api_key", batch_size=2)


@pytest.fixture
def ncbi_provider_no_key():
    return NCBIProvider(api_key=None, batch_size=2)
