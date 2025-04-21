# Component 3: Multi-layered Company Research Mapping

import os
import sys
import logging
import asyncio
import traceback
import streamlit as st
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from pydantic import BaseModel, Field
from datetime import datetime
from tavily import AsyncTavilyClient
from pydantic_ai import Agent, RunContext
from pydantic_ai.result import RunResult
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from cleanco import basename 
import re
import io
import time
import json
import math
import httpx
import nest_asyncio
nest_asyncio.apply()

def safe_async_run(coroutine):
    """
    Safely runs async coroutines, handling event loop management.
    
    Args:
        coroutine: The async coroutine to run
        
    Returns:
        The result of the coroutine
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        
        # Check if the loop is closed
        if loop.is_closed():
            # Create a new event loop if closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the coroutine
        return loop.run_until_complete(coroutine)
    except RuntimeError as e:
        # Handle various runtime errors
        if "There is no current event loop in thread" in str(e):
            # Create a new event loop if there is none
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coroutine)
        else:
            # Re-raise other RuntimeErrors
            raise


# Initialize Tavily client
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

# Logger setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# Initialize API credentials from Streamlit secrets
GEMINI_API_KEY = st.secrets["GOOGLE_AI_STUDIO"]

# Models setup
gemini_2o_model = GeminiModel('gemini-2.0-flash', api_key=GEMINI_API_KEY)
gemini_2o_thinking_model = GeminiModel('gemini-2.0-flash-thinking-exp-01-21', api_key=GEMINI_API_KEY)

# Select the model to use (can be changed to gemini_2o_thinking_model)
openrouter_model = gemini_2o_model
logger.info(f"Using model: {openrouter_model}")

# Initialize semaphore for managing concurrent API calls
sem = asyncio.Semaphore(1000)  # Limit to 1000 concurrent API calls to avoid rate limits

#########################################
# Core Cache Management Functions
#########################################
import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Cache configuration
CACHE_DIR = "tavily_search_result"
CACHE_EXPIRY_DAYS = 30  # Default cache expiry in days

def ensure_cache_dir(query_type: str) -> str:
    """
    Ensure the cache directory exists for the given query type.
    
    Args:
        query_type: Type of query (general, news, funding, entity)
        
    Returns:
        Path to the cache directory
    """
    cache_path = os.path.join(CACHE_DIR, query_type)
    os.makedirs(cache_path, exist_ok=True)
    return cache_path

def generate_cache_key(query: str) -> str:
    """
    Generate a unique cache key for a query using MD5 hash.
    """
    return hashlib.md5(query.encode()).hexdigest()

def get_cache_file_path(query: str, query_type: str) -> str:
    """
    Get the file path for a cached query result.
    """
    cache_dir = ensure_cache_dir(query_type)
    cache_key = generate_cache_key(query)
    return os.path.join(cache_dir, f"{cache_key}.json")

def save_to_cache(query: str, query_type: str, results: List[Dict], max_results: int, search_depth: str) -> None:
    """
    Save search results to cache with metadata.
    """
    cache_file = get_cache_file_path(query, query_type)
    
    # Prepare cache data with metadata
    cache_data = {
        "query": query,
        "query_type": query_type,
        "max_results": max_results,
        "search_depth": search_depth,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    # Save to file
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

def load_from_cache(query: str, query_type: str, max_results: int = None, search_depth: str = None) -> Optional[List[Dict]]:
    """
    Load search results from cache if available and not expired.
    """
    cache_file = get_cache_file_path(query, query_type)
    
    # Check if cache file exists
    if not os.path.exists(cache_file):
        return None
    
    try:
        # Load cache data
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Get timestamp from cache
        timestamp_str = cache_data.get("timestamp")
        if not timestamp_str:
            return None
        
        # Parse timestamp
        cache_timestamp = datetime.fromisoformat(timestamp_str)
        
        # Check if cache is expired
        if is_cache_expired(cache_timestamp):
            return None
        
        # If max_results or search_depth is specified, validate they match
        if max_results is not None and cache_data.get("max_results") != max_results:
            return None
        
        if search_depth is not None and cache_data.get("search_depth") != search_depth:
            return None
        
        # Return cached results
        return cache_data.get("results", [])
    
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
        return None

def is_cache_expired(timestamp: datetime) -> bool:
    """
    Check if a cache entry is expired.
    """
    expiry_date = timestamp + timedelta(days=CACHE_EXPIRY_DAYS)
    return datetime.now() > expiry_date

def clear_cache(query_type: Optional[str] = None) -> int:
    """
    Clear the search cache.
    
    Args:
        query_type: Optional type of query to clear (if None, clears all)
        
    Returns:
        Number of files deleted
    """
    deleted_count = 0
    
    if query_type:
        # Clear specific query type
        cache_dir = os.path.join(CACHE_DIR, query_type)
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(cache_dir, filename))
                    deleted_count += 1
    else:
        # Clear all query types
        if os.path.exists(CACHE_DIR):
            for subdir in os.listdir(CACHE_DIR):
                subdir_path = os.path.join(CACHE_DIR, subdir)
                if os.path.isdir(subdir_path):
                    for filename in os.listdir(subdir_path):
                        if filename.endswith(".json"):
                            os.remove(os.path.join(subdir_path, filename))
                            deleted_count += 1
    
    return deleted_count

def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the search cache.
    """
    stats = {
        "total_files": 0,
        "total_size_mb": 0,
        "by_type": {},
        "oldest_file": None,
        "newest_file": None
    }
    
    if not os.path.exists(CACHE_DIR):
        return stats
    
    oldest_timestamp = None
    newest_timestamp = None
    
    for subdir in os.listdir(CACHE_DIR):
        subdir_path = os.path.join(CACHE_DIR, subdir)
        if os.path.isdir(subdir_path):
            type_count = 0
            type_size = 0
            
            for filename in os.listdir(subdir_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(subdir_path, filename)
                    file_size = os.path.getsize(file_path)
                    file_time = os.path.getmtime(file_path)
                    
                    # Update counts and sizes
                    type_count += 1
                    type_size += file_size
                    stats["total_files"] += 1
                    stats["total_size_mb"] += file_size / (1024 * 1024)
                    
                    # Update timestamp tracking
                    if oldest_timestamp is None or file_time < oldest_timestamp:
                        oldest_timestamp = file_time
                        stats["oldest_file"] = datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
                    
                    if newest_timestamp is None or file_time > newest_timestamp:
                        newest_timestamp = file_time
                        stats["newest_file"] = datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
            
            # Store type statistics
            if type_count > 0:
                stats["by_type"][subdir] = {
                    "count": type_count,
                    "size_mb": round(type_size / (1024 * 1024), 2)
                }
    
    # Round total size
    stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    
    return stats

def add_cache_controls_to_sidebar():
    """Add cache control elements with credit exhaustion handling."""
    st.sidebar.subheader("Search Cache Controls")
    
    # Initialize cache usage stats if not present
    if 'cache_usage_stats' not in st.session_state:
        st.session_state.cache_usage_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_saved": 0,
            "time_saved": 0
        }
    
    # Initialize cache control settings if not present
    if 'use_cache' not in st.session_state:
        st.session_state['use_cache'] = True
    
    if 'force_refresh' not in st.session_state:
        st.session_state['force_refresh'] = False
    
    # Special handling if API credits are exhausted
    api_credits_exhausted = False
    if 'api_status' in st.session_state and not st.session_state.api_status.get('has_credits', True):
        api_credits_exhausted = True
        st.sidebar.warning("⚠️ Tavily API credits exhausted")
        st.sidebar.info("Working in cache-only mode")
        
        # Display cache settings but disable the controls
        st.sidebar.checkbox("Use cached search results", value=True, disabled=True, key="cache_control_1")
        st.sidebar.checkbox("Force refresh all searches", value=False, disabled=True, key="cache_control_2")
        
        # Force cache settings when credits are exhausted
        # Use dictionary-style access to avoid attribute error
        st.session_state['use_cache'] = True
        st.session_state['force_refresh'] = False
    else:
        # Normal operation - enable/disable cache checkbox
        use_cache = st.sidebar.checkbox("Use cached search results", value=st.session_state.get('use_cache', True), key="cache_control_1")
        
        # Force refresh checkbox - only enabled if cache is enabled
        force_refresh = st.sidebar.checkbox("Force refresh all searches", value=st.session_state.get('force_refresh', False), disabled=not use_cache, key="cache_control_2")
        
        # Update session state using dictionary-style access
        st.session_state['use_cache'] = use_cache
        st.session_state['force_refresh'] = force_refresh
    
    # Cache statistics
    stats = get_cache_stats()
    
    # Show cache stats if any files exist
    if stats["total_files"] > 0:
        st.sidebar.subheader("Cache Statistics")
        st.sidebar.write(f"Total files: {stats['total_files']}")
        st.sidebar.write(f"Total size: {stats['total_size_mb']} MB")
        
        if stats["by_type"]:
            st.sidebar.write("Cache by type:")
            for query_type, type_stats in stats["by_type"].items():
                st.sidebar.write(f"- {query_type}: {type_stats['count']} files ({type_stats['size_mb']} MB)")
        
        if stats["oldest_file"]:
            st.sidebar.write(f"Oldest cached item: {stats['oldest_file']}")
        
        if stats["newest_file"]:
            st.sidebar.write(f"Newest cached item: {stats['newest_file']}")
        
        # Add clear cache button - disable when credits are exhausted
        if st.sidebar.button("Clear Search Cache", disabled=api_credits_exhausted):
            deleted = clear_cache()
            st.sidebar.success(f"Cleared {deleted} cached search results")
            
            # Reset cache usage stats as well
            st.session_state.cache_usage_stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "bytes_saved": 0,
                "time_saved": 0
            }

def add_cache_usage_stats_to_sidebar():
    """
    Add cache usage statistics for the current session to the sidebar.
    """
    if 'cache_usage_stats' not in st.session_state:
        st.session_state.cache_usage_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_saved": 0,
            "time_saved": 0
        }
    
    stats = st.session_state.cache_usage_stats
    
    st.sidebar.markdown("### Session Cache Stats")
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Cache Hits", stats["cache_hits"])
    col2.metric("Cache Misses", stats["cache_misses"])
    
    # Only show cache hit rate if there are any requests
    total_requests = stats["cache_hits"] + stats["cache_misses"]
    if total_requests > 0:
        hit_rate = stats["cache_hits"] / total_requests * 100
        st.sidebar.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
    
    if stats["time_saved"] > 0:
        time_saved = stats["time_saved"]
        if time_saved > 60:
            formatted_time = f"{time_saved/60:.1f} minutes"
        else:
            formatted_time = f"{time_saved:.1f} seconds"
        st.sidebar.metric("Estimated Time Saved", formatted_time)
    
    if stats["bytes_saved"] > 0:
        bytes_saved = stats["bytes_saved"]
        if bytes_saved > 1024*1024:
            formatted_size = f"{bytes_saved/(1024*1024):.1f} MB"
        else:
            formatted_size = f"{bytes_saved/1024:.1f} KB"
        st.sidebar.metric("Estimated Bandwidth Saved", formatted_size)

#########################################
# Pydantic Models for Structured Data
#########################################

class ProductServiceDetails(BaseModel):
    """Detailed information about a product or service."""
    product_name: str = Field(..., description="The name of the product or service.")
    company: str = Field(..., description="URL of the company.")
    product_type: str = Field(..., description="The type of product or service.")
    scientific_domain: str = Field(..., description="The scientific domain the product/service belongs to.")
    accessibility: str = Field(..., description="Information about the accessibility or licensing of the product/service.")
    description_abstract: Optional[str] = Field(None, description="A brief description or abstract of the product/service.")
    publications_url: Optional[str] = Field(None, description="URL to relevant publications about the product/service.")

class ScientificDomainDetails(BaseModel):
    """Detailed information about scientific domains and industry classification."""
    primary_domain: str = Field(..., description="Primary scientific domain (e.g., 'Biotechnology', 'Artificial Intelligence').")
    sub_domains: List[str] = Field(default_factory=list, description="Specific sub-domains within the primary domain.")
    industry_classification: Optional[str] = Field(None, description="Industry classification (e.g., 'Healthcare', 'Finance').")
    technological_areas: List[str] = Field(default_factory=list, description="Key technological areas of focus.")
    scientific_approaches: List[str] = Field(default_factory=list, description="Scientific methodologies or approaches used.")
    innovation_areas: List[str] = Field(default_factory=list, description="Areas of innovation or research focus.")

class BusinessModelDetails(BaseModel):
    """Detailed information about business models and revenue streams."""
    primary_models: List[str] = Field(..., description="Primary business models (e.g., 'SaaS', 'Marketplace').")
    revenue_streams: List[str] = Field(default_factory=list, description="Sources of revenue.")
    customer_segments: List[str] = Field(default_factory=list, description="Target customer segments.")
    value_proposition: Optional[str] = Field(None, description="Core value proposition.")
    pricing_strategy: Optional[str] = Field(None, description="Pricing strategy or model.")
    go_to_market: Optional[str] = Field(None, description="Go-to-market strategy.")
    partnership_strategy: Optional[str] = Field(None, description="Partnership or channel strategy.")

class LabDataGeneration(BaseModel):
    """Input for lab or proprietary data generation information."""
    data_generation_types: List[Optional[str]] = Field(..., description="List of methods for lab or proprietary data generation.")

class DrugPipeline(BaseModel):
    """Input for drug pipeline stages."""
    pipeline_stages: List[Optional[str]] = Field(..., description="List of drug pipeline stages.")

class OrganizationType(BaseModel):
    """Input for the organization type."""
    organization_type: str = Field(..., description="The type of organization.")

class HQLocations(BaseModel):
    """Input for company HQ locations."""
    locations: List[str] = Field(..., description="List of HQ locations for different companies.")

class RelevantSegmentDetails(BaseModel):
    """Details of the relevant segments, tailored to the specific data provided."""
    segments: List[str] = Field(..., description="""
        List of relevant segments for the entity.
        Possible segments include:
        - 'Molecular Design'
        - 'Disease Biology'
        - 'Manufacturing'
        - 'Workflow / Data / DOE'
        - 'Materials'
        - 'Automation'
        - 'Biological Tools'
        - 'PK/PD/Tox'
        - 'AI Scientists'
        - 'Biosecurity'
        - 'Reporting'
        Segments can be combined, e.g., 'Disease Biology, Molecular Design'.
    """)

class FundingMarketCapBooleans(BaseModel):
    """Boolean indicators for estimated funding or market cap."""
    is_it_bootstrapped_low: bool = Field(False, description="Indicates if the funding/market cap is Bootstrapped / Low.")
    is_it_modest: bool = Field(False, description="Indicates if the funding/market cap is Modest (10s of $M).")
    is_it_mega: bool = Field(False, description="Indicates if the funding/market cap is Mega ($500M+).")
    is_it_significant: bool = Field(False, description="Indicates if the funding/market cap is Significant (low 100s of $M).")

class FundingStageBooleans(BaseModel):
    """Boolean indicators for the funding stage."""
    is_it_bootstrapped_stage: bool = Field(False, description="Indicates if the funding stage is Bootstrapped.")
    is_it_pre_seed: bool = Field(False, description="Indicates if the funding stage is Pre-seed.")
    is_it_seed: bool = Field(False, description="Indicates if the funding stage is Seed.")
    is_it_series_a: bool = Field(False, description="Indicates if the funding stage is Series A.")
    is_it_series_b: bool = Field(False, description="Indicates if the funding stage is Series B.")
    is_it_series_c: bool = Field(False, description="Indicates if the funding stage is Series C.")
    is_it_series_d: bool = Field(False, description="Indicates if the funding stage is Series D.")
    is_it_series_e: bool = Field(False, description="Indicates if the funding stage is Series E.")
    is_it_series_f: bool = Field(False, description="Indicates if the funding stage is Series F.")
    is_it_public: bool = Field(False, description="Indicates if the funding stage is Public.")
    is_it_donations_grant: bool = Field(False, description="Indicates if the funding stage is Donations / Grant.")
    is_it_acquired: bool = Field(False, description="Indicates if the funding stage is Acquired.")
    is_it_subsidiary: bool = Field(False, description="Indicates if the funding stage is Subsidiary.")

class FundingRoundDetails(BaseModel):
    """Detailed information about a funding round."""
    round_date: Optional[str] = Field(None, description="Date of the funding round (e.g., 'May 1, 2015').")
    round_type: Optional[str] = Field(None, description="Type of funding round (e.g., 'Seed', 'Series A').")
    amount: Optional[str] = Field(None, description="Amount raised in the funding round (e.g., '$2,100,000').")
    investors: Optional[List[str]] = Field(None, description="Investors participating in this round.")
    lead_investor: Optional[str] = Field(None, description="Lead investor in the funding round, if specified.")
    post_money_valuation: Optional[str] = Field(None, description="Post-money valuation after the round.")

class FundingRoundList(BaseModel):
    """Container for multiple funding rounds."""
    funding_rounds: List[FundingRoundDetails] = Field(..., description="Funding Round Details")
    total_funding: Optional[str] = Field(None, description="Total funding raised across all rounds.")
    total_rounds: Optional[int] = Field(None, description="Total number of funding rounds.")
    last_funding_date: Optional[str] = Field(None, description="Date of the most recent funding round.")

class YearFoundedDetails(BaseModel):
    """Details of the year the entity was founded."""
    year_founded: int = Field(..., description="The year the entity was founded.")

class KeyPeopleDetails(BaseModel):
    """Information about advisors, board members, or key people."""
    name: str = Field(..., description="Full name of the person.")
    role: str = Field(..., description="Specific role or title (e.g., Advisor, Board Member, CEO).")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL of the person.")
    twitter_url: Optional[str] = Field(None, description="Twitter profile URL of the person.")
    biography: Optional[str] = Field(None, description="Short biography or background information.")

class EnhancedKeyPeopleDetails(BaseModel):
    """Enhanced information about key people."""
    name: str = Field(..., description="Full name of the person.")
    role: str = Field(..., description="Specific role or title at the company.")
    is_founder: bool = Field(False, description="Whether this person is a founder.")
    is_executive: bool = Field(False, description="Whether this person is a C-level executive.")
    is_board_member: bool = Field(False, description="Whether this person is a board member.")
    is_advisor: bool = Field(False, description="Whether this person is an advisor.")
    biography: Optional[str] = Field(None, description="Brief biography or background.")
    previous_positions: List[str] = Field(default_factory=list, description="Previous notable positions.")
    education: Optional[str] = Field(None, description="Educational background.")
    expertise_areas: List[str] = Field(default_factory=list, description="Areas of expertise or specialization.")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL.")
    twitter_url: Optional[str] = Field(None, description="Twitter profile URL.")
    joined_date: Optional[str] = Field(None, description="When the person joined the company.")

class CompetitorDetails(BaseModel):
    """Details about a competitor."""
    company_name: str = Field(..., description="Name of the competitor company.")
    relationship: str = Field(..., description="Relationship to target company (e.g., 'Direct Competitor', 'Indirect Competitor').")
    company_focus: Optional[str] = Field(None, description="Business focus of the competitor.")
    competitive_products: List[str] = Field(default_factory=list, description="Competing products or services.")
    key_strengths: List[str] = Field(default_factory=list, description="Key strengths of the competitor.")
    key_weaknesses: List[str] = Field(default_factory=list, description="Key weaknesses of the competitor.")
    market_position: Optional[str] = Field(None, description="Market position relative to target company.")
    differentiating_factors: List[str] = Field(default_factory=list, description="How competitor differentiates from target company.")

class CompetitiveLandscape(BaseModel):
    """Structured competitive landscape analysis."""
    target_company: str = Field(..., description="Name of the target company being analyzed.")
    market_sector: str = Field(..., description="Market sector or industry being analyzed.")
    direct_competitors: List[CompetitorDetails] = Field(default_factory=list, description="Direct competitors in the same space.")
    indirect_competitors: List[CompetitorDetails] = Field(default_factory=list, description="Indirect competitors in adjacent spaces.")
    market_leaders: List[str] = Field(default_factory=list, description="Market leaders in this space.")
    competitive_positioning: Optional[str] = Field(None, description="Target company's competitive positioning.")
    target_advantages: List[str] = Field(default_factory=list, description="Target company's competitive advantages.")
    target_challenges: List[str] = Field(default_factory=list, description="Target company's competitive challenges.")
    market_trends: List[str] = Field(default_factory=list, description="Key trends affecting the competitive landscape.")

class InvestorDetails(BaseModel):
    """Detailed information about an investor."""
    investor_name: str = Field(..., description="Name of the investor or investment firm.")
    investor_type: Optional[str] = Field(None, description="Type of investor (e.g., Venture Capital, Angel Investor, Private Equity).")
    investment_stage: Optional[str] = Field(None, description="Typical investment stage focus (e.g., Seed, Series A, Growth).")
    location: Optional[str] = Field(None, description="Geographic location of the investor or firm.")
    website: Optional[str] = Field(None, description="Website of the investor or firm.")
    description: Optional[str] = Field(None, description="Brief description of the investor's focus or strategy.")
    portfolio_companies: Optional[List[str]] = Field(None, description="List of portfolio companies (optional).")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL of the investor or firm.")
    twitter_url: Optional[str] = Field(None, description="Twitter profile URL of the investor or firm.")
    focus_areas: Optional[List[str]] = Field(None, description="Specific industries or areas the investor focuses on.")

class InvestorList(BaseModel):
    """Container for multiple investors."""
    investors: List[InvestorDetails] = Field(..., description="Investor Details")

class PartnershipDetails(BaseModel):
    """Details of a partnership."""
    company_name: Optional[str] = Field(None, alias="Company Name", description="Name of the company.")
    partnership_entities: Optional[str] = Field(None, alias="Partnership entities", description="Entities involved in the partnership.")
    url: Optional[str] = Field(None, alias="URL", description="URL related to the partnership.")
    url_title: Optional[str] = Field(None, alias="URL Title", description="Title of the URL.")
    partnership_descriptions: Optional[str] = Field(None, alias="URL Summary or Partnership descriptions", description="Summary or description of the partnership.")
    year_it_happened: Optional[str] = Field(None, alias="Year it happened", description="Year the partnership happened.")
    deal_size: Optional[str] = Field(None, alias="Deal Size", description="Size of the deal.")
    royalties_or_equities: Optional[str] = Field(None, alias="Any Royalties or Equities?", description="Whether there are any royalties or equities involved.")

class EnhancedKeyPeopleDetails(BaseModel):
    """Enhanced information about key people."""
    name: str = Field(..., description="Full name of the person.")
    role: str = Field(..., description="Specific role or title at the company.")
    is_founder: bool = Field(False, description="Whether this person is a founder.")
    is_executive: bool = Field(False, description="Whether this person is a C-level executive.")
    is_board_member: bool = Field(False, description="Whether this person is a board member.")
    is_advisor: bool = Field(False, description="Whether this person is an advisor.")
    biography: Optional[str] = Field(None, description="Brief biography or background.")
    previous_positions: List[str] = Field(default_factory=list, description="Previous notable positions.")
    education: Optional[str] = Field(None, description="Educational background.")
    expertise_areas: List[str] = Field(default_factory=list, description="Areas of expertise or specialization.")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL.")
    twitter_url: Optional[str] = Field(None, description="Twitter profile URL.")
    joined_date: Optional[str] = Field(None, description="When the person joined the company.")

class FemaleCoFounder(BaseModel):
    """Data about co-founders and their gender."""
    female_co_founder: bool = Field(..., description="Whether there is a female co-founder (True for Yes, False for No).")

class WatchlistItem(BaseModel):
    """A single item on the watchlist."""
    description: str = Field(..., description="The emoji and description of the watchlist item.")

# Field-specific agent extraction functions
products_services_agent = Agent(
    model=gemini_2o_model,
    result_type=ProductServiceDetails,
    system_prompt="""You are an expert analyst identifying product and service details.
Extract and return information about products and services mentioned in the text, following the 'ProductServiceDetails' schema.
Focus on the name of the product or service, the company associated with it, its type, scientific domain, accessibility, a brief description, and URLs to relevant publications."""
)

async def extract_product_service_details(content: str) -> Optional[ProductServiceDetails]:
    """
    Extracts product/service details from content.
    Returns the model object directly rather than in a RunResult wrapper.
    """
    try:
        result = await products_services_agent.run(content)
        # Return the data directly - the model object itself
        return result.data
    except Exception as e:
        logger.error(f"Error extracting Product/Service details: {e}")
        return None

partnerships_agent = Agent(
    model=gemini_2o_model,
    result_type=PartnershipDetails,
    system_prompt="""You are an expert in identifying publicly announced partnerships.
Extract details of partnerships mentioned in the text and return them according to the 'PartnershipDetails' schema.
Include company names, entities involved, related URLs, partnership descriptions, dates, deal sizes, and information on royalties or equities."""
)

async def extract_partnership_details(content: str) -> Optional[PartnershipDetails]:
    try:
        result = await partnerships_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Partnership details: {e}")
        return None

lab_data_generation_agent = Agent(
    model=gemini_2o_model,
    result_type=LabDataGeneration,
    system_prompt="""You are identifying methods of lab or proprietary data generation.
Extract the methods or types of lab or proprietary data generation mentioned in the text, following the 'LabDataGeneration' schema."""
)

async def extract_lab_data_generation(content: str) -> Optional[LabDataGeneration]:
    try:
        result = await lab_data_generation_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Lab Data Generation information: {e}")
        return None

drug_pipeline_agent = Agent(
    model=gemini_2o_model,
    result_type=DrugPipeline,
    system_prompt="""You are an expert on drug development pipelines.
Extract the stages of the drug pipeline mentioned in the text, adhering to the 'DrugPipeline' schema."""
)

async def extract_drug_pipeline(content: str) -> Optional[DrugPipeline]:
    try:
        result = await drug_pipeline_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Drug Pipeline information: {e}")
        return None

organization_type_agent = Agent(
    model=gemini_2o_model,
    result_type=OrganizationType,
    system_prompt="""You are identifying the type of organization.
Extract and return the organization type mentioned in the text, following the 'OrganizationType' schema."""
)

async def extract_organization_type(content: str) -> Optional[OrganizationType]:
    try:
        result = await organization_type_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Organization Type: {e}")
        return None

hq_location_agent = Agent(
    model=gemini_2o_model,
    result_type=HQLocations,
    system_prompt="""You are tasked with identifying company headquarters locations.
Extract and list the HQ locations mentioned in the text, adhering to the 'HQLocations' schema."""
)

async def extract_hq_locations(content: str) -> Optional[HQLocations]:
    try:
        result = await hq_location_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting HQ Locations: {e}")
        return None

relevant_segments_agent = Agent(
    model=gemini_2o_model,
    result_type=RelevantSegmentDetails,
    system_prompt="""You are an expert in identifying relevant business segments.
Extract the relevant business segments mentioned in the text, ensuring they align with the examples provided in the 'RelevantSegmentDetails' schema."""
)

async def extract_relevant_segments(content: str) -> Optional[RelevantSegmentDetails]:
    try:
        result = await relevant_segments_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Relevant Segments: {e}")
        return None

funding_stage_agent = Agent(
    model=gemini_2o_model,
    result_type=FundingStageBooleans,
    system_prompt="""You are identifying the funding stage of an entity.
For each funding stage (Bootstrapped, Pre-seed, Seed, Series A, Series B, Series C, Series D, Series E, Series F, Public, Donations / Grant, Acquired, Subsidiary), determine if it matches the provided text.
Return True if the stage is mentioned or implied, and False otherwise. Adhere to the 'FundingStageBooleans' schema."""
)

async def extract_funding_stage(content: str) -> Optional[FundingStageBooleans]:
    try:
        result = await funding_stage_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Funding Stage: {e}")
        return None

funding_market_cap_agent = Agent(
    model=gemini_2o_model,
    result_type=FundingMarketCapBooleans,
    system_prompt="""You are identifying the estimated funding or market capitalization level of an entity.
For each category (Bootstrapped / Low, Modest (10s of $M), Mega ($500M+), Significant (low 100s of $M)), determine if it matches the provided text.
Return True if the category is mentioned or implied, and False otherwise. Adhere to the 'FundingMarketCapBooleans' schema."""
)

funding_rounds_agent = Agent(
    model=gemini_2o_model,
    result_type=FundingRoundList,
    system_prompt="""You are an expert at extracting structured funding round information from text.

Extract details of each funding round mentioned in the text following the FundingRoundList schema.
Focus on accurately capturing:

1. Exact funding dates (e.g., "May 1, 2015")
2. Precise round types (e.g., "Seed", "Series A", "Series B") 
3. Complete funding amounts with currency symbols (e.g., "$2,100,000")
4. All investors involved in each round (e.g., "Blumberg Capital and 4 other investors")

When extracting investors:
- Include the exact names as mentioned in the text
- Preserve information about how many other unnamed investors participated
- Identify lead investors when explicitly mentioned

For each funding round, provide clear standardized formatting:
- Dates as "Month Day, Year" (e.g., "May 1, 2015")
- Amounts with currency symbols (e.g., "$2,100,000")
- Round types with proper capitalization (e.g., "Seed", "Series A")

Also extract summary information:
- Total funding across all rounds
- Total number of funding rounds mentioned
- Details of the most recent funding round

Organize funding rounds chronologically with the most recent first."""
)

async def extract_funding_rounds(content: str) -> Optional[FundingRoundList]:
    """
    Extracts structured funding round information from content text.
    
    Args:
        content: The text content to extract from
        
    Returns:
        FundingRoundList with structured funding round information
    """
    try:
        result = await funding_rounds_agent.run(content)
        return result.data
    except Exception as e:
        logger.error(f"Error extracting Funding Round details: {e}")
        return None

async def extract_funding_market_cap(content: str) -> Optional[FundingMarketCapBooleans]:
    try:
        result = await funding_market_cap_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Funding/Market Cap information: {e}")
        return None

year_founded_agent = Agent(
    model=gemini_2o_model,
    result_type=YearFoundedDetails,
    system_prompt="""You are tasked with identifying the year an entity was founded.
Extract the year founded mentioned in the text, following the 'YearFoundedDetails' schema."""
)

async def extract_year_founded(content: str) -> Optional[YearFoundedDetails]:
    try:
        result = await year_founded_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Year Founded: {e}")
        return None

key_people_agent = Agent(
    model=gemini_2o_model,
    result_type=EnhancedKeyPeopleDetails,
    system_prompt="""You are an expert at extracting detailed information about key people in organizations.

Extract comprehensive information about a person mentioned in the text, following the 'EnhancedKeyPeopleDetails' schema.
Focus on extracting:

1. Full name and exact role/title at the company
2. Whether they are a founder, executive, board member, or advisor
3. Biographical information and background
4. Previous positions and career history
5. Educational background
6. Areas of expertise or specialization
7. Social media profiles (LinkedIn, Twitter)
8. When they joined the company (if mentioned)

Ensure accuracy in categorization:
- Mark someone as a founder only if explicitly described as such
- Identify executives based on C-level or VP-level titles
- Categorize board members based on explicit board membership mentions
- Identify advisors when explicitly described in an advisory capacity

Extract as much detail as available in the text, but don't make assumptions about information not present.
Maintain the exact spelling of names and titles as they appear in the text.
"""
)

async def extract_enhanced_key_people_details(content: str) -> Optional[EnhancedKeyPeopleDetails]:
    """
    Extracts enhanced key people details from content with improved error handling.
    
    Args:
        content: The text content to extract from
        
    Returns:
        EnhancedKeyPeopleDetails with comprehensive information about key people
    """
    try:
        result = await key_people_agent.run(content)
        return result.data
    except Exception as e:
        logger.error(f"Error extracting Enhanced Key People details: {e}")
        return None

async def extract_key_people_details(content: str) -> Optional[KeyPeopleDetails]:
    try:
        result = await key_people_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Key People details: {e}")
        return None

competitive_landscape_agent = Agent(
    model=gemini_2o_model,
    result_type=CompetitiveLandscape,
    system_prompt="""You are an expert at analyzing competitive landscapes for companies.

Extract detailed information about a company's competitive positioning, following the 'CompetitiveLandscape' schema.
Focus on extracting:

1. The target company and its market sector
2. Direct competitors in the same space with their details
3. Indirect competitors in adjacent spaces
4. Market leaders in the sector
5. Competitive positioning of the target company
6. Target company's competitive advantages and challenges
7. Key market trends affecting the competitive landscape

For each competitor identified, extract:
- Company name
- Relationship to target (direct/indirect)
- Business focus
- Competing products or services
- Key strengths and weaknesses
- Market position
- Differentiating factors

Extract only information clearly present in the text without making assumptions.
Focus specifically on competitive dynamics rather than general company information.
Maintain precision in categorizing competitors as direct or indirect based on their market overlap.
"""
)

async def extract_competitive_landscape(content: str) -> Optional[CompetitiveLandscape]:
    """
    Extracts competitive landscape information from content.
    
    Args:
        content: The text content to extract from
        
    Returns:
        CompetitiveLandscape with structured competitive analysis
    """
    try:
        result = await competitive_landscape_agent.run(content)
        return result.data
    except Exception as e:
        logger.error(f"Error extracting Competitive Landscape: {e}")
        return None

scientific_domain_agent = Agent(
    model=gemini_2o_model,
    result_type=ScientificDomainDetails,
    system_prompt="""You are an expert at identifying scientific domains and technological areas.

Extract detailed information about scientific domains and technological focus areas from the text, following the 'ScientificDomainDetails' schema.
Focus on extracting:

1. Primary scientific domain (e.g., Biotechnology, Artificial Intelligence)
2. Specific sub-domains within the primary domain
3. Industry classification (e.g., Healthcare, Finance)
4. Key technological areas of focus
5. Scientific methodologies or approaches used
6. Areas of innovation or research focus

Be precise in domain classification based on scientific and technical content.
Differentiate between broad domains and specific sub-domains.
Identify technological areas based on specific technologies mentioned.
Extract scientific approaches based on methodologies and techniques described.
Identify innovation areas based on research focus and development priorities.

Extract information directly from the text without making assumptions.
Use exact terminology as presented in the content when possible.
"""
)

async def extract_scientific_domain_details(content: str) -> Optional[ScientificDomainDetails]:
    """
    Extracts scientific domain details from content.
    
    Args:
        content: The text content to extract from
        
    Returns:
        ScientificDomainDetails with structured scientific domain information
    """
    try:
        result = await scientific_domain_agent.run(content)
        return result.data
    except Exception as e:
        logger.error(f"Error extracting Scientific Domain details: {e}")
        return None

business_model_agent = Agent(
    model=gemini_2o_model,
    result_type=BusinessModelDetails,
    system_prompt="""You are an expert at identifying business models and revenue strategies.

Extract detailed information about business models and revenue approaches from the text, following the 'BusinessModelDetails' schema.
Focus on extracting:

1. Primary business models (e.g., SaaS, Marketplace, Subscription)
2. Revenue streams and monetization approaches
3. Target customer segments
4. Core value proposition
5. Pricing strategy or model
6. Go-to-market strategy
7. Partnership or channel strategy

Be precise in categorizing business models using standard terminology:
- SaaS (Software as a Service)
- PaaS (Platform as a Service)
- IaaS (Infrastructure as a Service)
- Subscription
- Marketplace
- E-commerce
- Licensing
- Advertising
- Freemium
- Service-based

Identify revenue streams as distinct sources of income.
Categorize customer segments based on industry, size, or characteristics.
Extract value proposition based on core benefits or solutions provided.
Identify pricing approach (e.g., tiered, usage-based, flat-rate).
Extract go-to-market and partnership strategies when mentioned.

Extract only information explicitly mentioned without speculation.
"""
)

async def extract_business_model_details(content: str) -> Optional[BusinessModelDetails]:
    """
    Extracts business model details from content.
    
    Args:
        content: The text content to extract from
        
    Returns:
        BusinessModelDetails with structured business model information
    """
    try:
        result = await business_model_agent.run(content)
        return result.data
    except Exception as e:
        logger.error(f"Error extracting Business Model details: {e}")
        return None

investors_agent = Agent(
    model=gemini_2o_model,
    result_type=InvestorList,
    system_prompt="""You are an expert at identifying investors and investment firms in company funding contexts.

Extract details of investors mentioned in the text, following the 'InvestorList' schema.
Focus specifically on:

1. Exact investor names as they appear in the text
2. The funding rounds they participated in (with dates when available)
3. Whether they were lead investors
4. Investor types (VC, Angel, PE, etc.)
5. Investment focus areas or stages

When extracting investor information:
- Link investors to specific funding rounds whenever possible
- Include dates of investment when mentioned (e.g., "May 1, 2015")
- Note the round type associated with each investment (e.g., "Seed", "Series A")
- Preserve information about "other investors" when mentioned generically

Format investor entries consistently:
- Include round information with each investor when available
- Standardize investor types (e.g., "Venture Capital", "Angel Investor")
- Preserve the exact spelling of investor names

Each investor should have a complete profile with all available information organized clearly.
"""
)

async def extract_investor_details(content: str) -> Optional[InvestorList]:
    """
    Extracts investor details from content.
    
    Args:
        content: The text content to extract from
        
    Returns:
        InvestorList with structured investor information
    """
    try:
        result = await investors_agent.run(content)
        return result.data
    except Exception as e:
        logger.error(f"Error extracting Investor details: {e}")
        return None

female_co_founder_agent = Agent(
    model=gemini_2o_model,
    result_type=FemaleCoFounder,
    system_prompt="""You are determining if there is a female co-founder.
Determine from the text if there is a female co-founder. Return True if yes, False if no, according to the 'FemaleCoFounder' schema."""
)

async def extract_female_co_founder(content: str) -> Optional[FemaleCoFounder]:
    try:
        result = await female_co_founder_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Female Co-Founder information: {e}")
        return None

watchlist_agent = Agent(
    model=gemini_2o_model,
    result_type=WatchlistItem,
    system_prompt="""You are identifying items for a watchlist.
Extract any descriptions or items that would be suitable for a watchlist, following the 'WatchlistItem' schema."""
)

async def extract_watchlist_item(content: str) -> Optional[WatchlistItem]:
    try:
        result = await watchlist_agent.run(content)
        return result.data
    except Exception as e:
        print(f"Error extracting Watchlist item: {e}")
        return None


########################################################################################################################################################################################

class BatchProcessTracker:
    """
    Manages real-time progress tracking for batch processing of company data.
    Now includes error handling for UI components.
    """
    def __init__(self, companies):
        self.companies = companies
        self.total_companies = len(companies)
        self.current_company_idx = 0
        self.steps_per_company = 5
        self.total_steps = self.total_companies * self.steps_per_company
        self.completed_steps = 0
        
        # UI containers - initialize with safe defaults
        self.progress_bar = None
        self.overall_status = None
        self.company_status = None
        self.step_status = None
        self.company_progress = {}
        self.company_progress_container = None
        
        # Enhanced search data tracking
        self.detailed_data = {}
        self.search_data = {}
        self.dataframe_container = None
        self.search_results_container = None
        self.current_dataframe = None
        self.dataframe_placeholder = None
        
        # Step timing data for estimates
        self.step_times = {}
        self.step_start_time = None
        self.start_time = None
        
        # UI initialization flag
        self.ui_initialized = False
    
    def initialize_ui(self):
        """Creates optimized UI components for tracking progress with improved container isolation."""
        # Create a main container for ALL tracker UI elements
        with st.container():
            st.subheader("Batch Processing Progress")
            self.progress_bar = st.progress(0.0)
            col1, col2 = st.columns(2)
            
            with col1:
                self.overall_status = st.empty()
                self.company_status = st.empty()
            
            with col2:
                self.step_status = st.empty()
                self.eta = st.empty()
            
            # Style definitions
            st.markdown("""
            <style>
            .progress-header {
                background-color: #f0f2f6;
                padding: 5px 10px;
                border-radius: 5px 5px 0 0;
                margin-bottom: 10px;
                font-weight: bold;
            }
            .section-container {
                border: 1px solid #dddddd;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create explicit containers with clear scope
            progress_section = st.container()
            dataframe_section = st.container()
            
            # 1. Company Progress Section - REPLACED NESTED EXPANDERS WITH DATAFRAME
            with progress_section:
                st.markdown("<div class='progress-header'>Detailed Company Search Progress</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-container'>", unsafe_allow_html=True)
                
                # Create a specific container just for company progress
                self.company_progress_container = st.container()
                
                # Initialize company progress using a DATAFRAME INSTEAD OF EXPANDERS
                with self.company_progress_container:
                    progress_data = {
                        "Company": [],
                        "Status": [],
                        "Progress": [],
                        "Current Step": []
                    }
                    
                    for idx, company in enumerate(self.companies):
                        company_name = company["name"]
                        progress_data["Company"].append(company_name)
                        progress_data["Status"].append("Pending")
                        progress_data["Progress"].append(0.0)
                        progress_data["Current Step"].append("Not Started")
                        
                        # Initialize tracking data structures
                        self.company_progress[company_name] = {
                            "progress": 0.0,
                            "status": "Pending",
                            "last_update": None,
                            "current_step": "Not Started",
                            "log_messages": []  # Store log messages instead of using nested expanders
                        }
                    
                    # Display as interactive dataframe
                    self.progress_df = pd.DataFrame(progress_data)
                    self.progress_table = st.dataframe(self.progress_df, use_container_width=True)
                    
                    # Add a separate container for detailed logs of selected company
                    st.subheader("Activity Log")
                    company_selector = st.selectbox(
                        "Select company to view detailed logs:",
                        options=[c["name"] for c in self.companies],
                        key="company_log_selector"
                    )
                    
                    self.log_container = st.container()
                    self.selected_company_log = company_selector
                    
                    # Initialize data tracking structures
                    for company_name in [c["name"] for c in self.companies]:
                        self.detailed_data[company_name] = []
                        self.search_data[company_name] = {}
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # 2. Live Operations DataFrame Section
            with dataframe_section:
                st.markdown("<div class='progress-header'>Live Search and Extraction Data</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-container'>", unsafe_allow_html=True)
                
                self.dataframe_container = st.container()
                with self.dataframe_container:
                    self.dataframe_placeholder = st.empty()
                    self._initialize_dataframe()
                
                st.markdown("</div>", unsafe_allow_html=True)
          
            
            self.start_time = time.time()
            self.step_start_time = time.time()

    # Additional method to update company log display
    def update_company_log_display(self):
        """Updates the log display for the currently selected company."""
        if hasattr(self, 'log_container') and hasattr(self, 'selected_company_log'):
            with self.log_container:
                self.log_container.empty()
                company_name = self.selected_company_log
                
                if company_name in self.company_progress:
                    log_messages = self.company_progress[company_name].get("log_messages", [])
                    
                    if log_messages:
                        for message in log_messages:
                            if message.get("type") == "error":
                                st.error(message["text"])
                            elif message.get("type") == "warning":
                                st.warning(message["text"])
                            elif message.get("type") == "success":
                                st.success(message["text"])
                            else:
                                st.info(message["text"])
                    else:
                        st.info(f"No activity logged for {company_name} yet.")
    
    def _create_fallback_ui(self):
        """Creates minimal UI components as fallback in case of initialization errors."""
        try:
            st.warning("Using fallback UI due to initialization error.")
            self.overall_status = st.empty()
            self.company_status = st.empty()
            self.step_status = st.empty()
            self.progress_bar = st.progress(0.0)
            self.dataframe_placeholder = st.empty()
            self.ui_initialized = True
        except Exception as e:
            logger.error(f"Error creating fallback UI: {str(e)}")
            # Create dummy placeholders that won't throw errors
            self._create_dummy_components()
    
    def _create_dummy_components(self):
        """Creates dummy components that support required methods but don't use Streamlit."""
        class DummyComponent:
            def __init__(self):
                pass
            
            def info(self, text):
                logger.info(text)
                
            def warning(self, text):
                logger.warning(text)
                
            def error(self, text):
                logger.error(text)
                
            def success(self, text):
                logger.info(f"SUCCESS: {text}")
                
            def progress(self, value):
                pass
                
            def empty(self):
                return self
                
            def dataframe(self, df, **kwargs):
                pass
        
        # Replace UI components with dummy versions
        self.overall_status = DummyComponent()
        self.company_status = DummyComponent()
        self.step_status = DummyComponent()
        self.progress_bar = DummyComponent()
        self.dataframe_placeholder = DummyComponent()
        self.ui_initialized = True
    
    def update_status(self, message, status_type="info"):
        """
        Safely updates status message with fallback mechanisms.
        
        Args:
            message: Status message to display
            status_type: Type of status (info, warning, error, success)
        """
        # Ensure UI is initialized
        if not self.ui_initialized:
            logger.warning("Tracker UI not initialized. Initializing now.")
            self._create_fallback_ui()
        
        # Update status based on type and available components
        try:
            if self.overall_status is not None:
                if status_type == "info":
                    self.overall_status.info(message)
                elif status_type == "warning":
                    self.overall_status.warning(message)
                elif status_type == "error":
                    self.overall_status.error(message)
                elif status_type == "success":
                    self.overall_status.success(message)
            else:
                # Fallback to logging if component is unavailable
                logger.info(f"STATUS ({status_type}): {message}")
        except Exception as e:
            logger.error(f"Error updating status: {str(e)}")
            logger.info(f"STATUS ({status_type}): {message}")
    
    # Modify other status update methods to use update_status
    def overall_status_info(self, message):
        """Safe wrapper for overall_status.info()"""
        self.update_status(message, "info")
    
    def overall_status_warning(self, message):
        """Safe wrapper for overall_status.warning()"""
        self.update_status(message, "warning")
    
    def overall_status_error(self, message):
        """Safe wrapper for overall_status.error()"""
        self.update_status(message, "error")
    
    def overall_status_success(self, message):
        """Safe wrapper for overall_status.success()"""
        self.update_status(message, "success")

    
    def _initialize_dataframe(self):
        """Initialize an empty dataframe with the enhanced columns for search tracking"""
        # Create empty dataframe with expanded columns
        self.current_dataframe = pd.DataFrame(columns=[
            "Timestamp", 
            "Company", 
            "Operation", 
            "Status", 
            "Details", 
            "URL",
            "Content Preview",
            "Validation Result",
            "Search Query",    
            "Result Count",    
            "Duration (s)",
            "Source Type"  # If this was valuable information from the explorer
        ])
        
        # Display the empty dataframe in the placeholder
        self.dataframe_placeholder.dataframe(self.current_dataframe, use_container_width=True)
    
    def add_data_entry(self, company_name, operation, status, details="", url="", 
                    content_preview="", validation_result="", search_query="", 
                    result_count=None, duration=None):
        """
        Enhanced method to add a new entry to the detailed data tracking and update the live dataframe
        with comprehensive search information.
        """
        # Create entry with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        entry = {
            "Timestamp": timestamp,
            "Company": company_name,
            "Operation": operation,
            "Status": status,
            "Details": details[:100] + "..." if details and len(details) > 100 else details,
            "URL": url,
            "Content Preview": content_preview[:100] + "..." if content_preview and len(content_preview) > 100 else content_preview,
            "Validation Result": validation_result,
            "Search Query": search_query,
            "Result Count": result_count,
            "Duration (s)": f"{duration:.2f}" if duration is not None else ""
        }
        
        # Add to detailed data - ensure the key exists first
        if company_name not in self.detailed_data:
            self.detailed_data[company_name] = []
        self.detailed_data[company_name].append(entry)
        
        # Add to dataframe and update display
        new_df = pd.DataFrame([entry])
        self.current_dataframe = pd.concat([new_df, self.current_dataframe], ignore_index=True)
        
        # Limit dataframe size to avoid performance issues
        if len(self.current_dataframe) > 1000:
            self.current_dataframe = self.current_dataframe.iloc[:1000]
        
        # Update the displayed dataframe
        if self.dataframe_placeholder:
            self.dataframe_placeholder.dataframe(self.current_dataframe, use_container_width=True)
    
    def start_company(self, company_idx):
        """Marks the start of processing for a specific company."""
        self.current_company_idx = company_idx
        company_name = self.companies[company_idx]["name"]
        
        # Update company status
        self.company_progress[company_name]["status"] = "Processing"
        self.company_status.info(f"Processing company {company_idx+1}/{self.total_companies}: **{company_name}**")
        
        # Update within the expander
        with self.company_progress[company_name]["container"]:
            st.markdown(f"**Status**: 🔄 Processing")
            
        # Add entry to detailed data tracking
        self.add_data_entry(
            company_name=company_name,
            operation="Start Processing",
            status="In Progress",
            details=f"Starting processing for company {company_idx+1}/{self.total_companies}"
        )
        
        # Update the last update time
        self.company_progress[company_name]["last_update"] = time.time()
        
        return company_name
    
    def update_step(self, step_name, company_name=None):
        """Updates the current processing step for a company."""
        if not company_name and self.current_company_idx < len(self.companies):
            company_name = self.companies[self.current_company_idx]["name"]
        
        # Update step timing data
        now = time.time()
        elapsed = now - self.step_start_time
        self.step_times[step_name] = self.step_times.get(step_name, []) + [elapsed]
        self.step_start_time = now
        
        # Update step status
        self.step_status.info(f"Step: **{step_name}**")
        
        # Update company progress if applicable
        if company_name in self.company_progress:
            # Calculate per-company progress (each step is worth 20%)
            step_idx = {"Search": 0, "Extract": 1, "Verify": 2, "Field Search": 3, "Compile": 4}.get(step_name, 0)
            progress = min(1.0, (step_idx + 1) / self.steps_per_company)
            
            self.company_progress[company_name]["progress"] = progress
            self.company_progress[company_name]["current_step"] = step_name
            
            # Add a timestamp to the log messages
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = {
                "timestamp": timestamp,
                "text": f"**{timestamp}**: {step_name} in progress...",
                "type": "info"
            }
            self.company_progress[company_name]["log_messages"].append(log_message)
            
            # Update dataframe without expanders
            if hasattr(self, 'progress_df'):
                company_idx = self.progress_df.index[self.progress_df["Company"] == company_name].tolist()
                if company_idx:
                    self.progress_df.loc[company_idx[0], "Progress"] = progress * 100  # Show as percentage
                    self.progress_df.loc[company_idx[0], "Current Step"] = step_name
                    self.progress_table.dataframe(self.progress_df, use_container_width=True)
            
            # Update selected company log if this is the selected company
            if hasattr(self, 'selected_company_log') and self.selected_company_log == company_name:
                self.update_company_log_display()
            
            # Add entry to detailed data tracking
            self.add_data_entry(
                company_name=company_name,
                operation=f"Step: {step_name}",
                status="In Progress",
                details=f"Starting step {step_idx+1}/{self.steps_per_company}",
                duration=elapsed
            )
        
        # Increment completed steps
        self.completed_steps += 1
        overall_progress = min(1.0, self.completed_steps / self.total_steps)
        self.progress_bar.progress(overall_progress)
        
        # Calculate ETA
        self._update_eta()
        
        # Store progress in session state
        st.session_state.batch_tracker = {
            'current_company': company_name,
            'current_step': step_name,
            'overall_progress': overall_progress,
            'completed_steps': self.completed_steps,
            'total_steps': self.total_steps
        }
    
    def log_search_query(self, company_name, query, start_time=None):
        """Enhanced method to log a search query execution with comprehensive tracking"""
        elapsed = time.time() - start_time if start_time else None
        
        # Initialize query results tracking if not already present
        if company_name in self.search_data:
            if query not in self.search_data[company_name]:
                self.search_data[company_name][query] = []
        else:
            self.search_data[company_name] = {query: []}
        
        # Add entry to detailed data tracking
        self.add_data_entry(
            company_name=company_name,
            operation="Search Query",
            status="Executed",
            details=query,
            search_query=query,
            duration=elapsed
        )
            
    def log_search_result(self, company_name, result, query=""):
        """Enhanced method to log a search result with full result storage and visualization"""
        # Extract information from the result
        title = result.get('title', 'No title')
        url = result.get('url', '')
        content = result.get('content', '')
        
        # Store the complete result in search data structure
        if company_name in self.search_data and query in self.search_data[company_name]:
            self.search_data[company_name][query].append(result)
            result_count = len(self.search_data[company_name][query])
        else:
            if company_name not in self.search_data:
                self.search_data[company_name] = {}
            if query not in self.search_data[company_name]:
                self.search_data[company_name][query] = []
            self.search_data[company_name][query].append(result)
            result_count = 1
        
        # Add entry to detailed data tracking
        self.add_data_entry(
            company_name=company_name,
            operation="Search Result",
            status="Found",
            details=f"Title: {title} (from query: {query})",
            url=url,
            content_preview=content,
            search_query=query,
            result_count=result_count
        )
                    
    def log_content_validation(self, company_name, url, validation_result, content_preview="", query=""):
        """Enhanced method to log content validation with query tracking"""
        # Extract validation information
        is_valid = validation_result.get('is_valid', False)
        confidence = validation_result.get('confidence', 'Unknown')
        reason = validation_result.get('reason', '')
        content_type = validation_result.get('content_type', '')
        
        status = "Valid" if is_valid else "Invalid"
        details = f"{content_type}: {reason}" if not is_valid else "Content validated successfully"
        
        # Find the result in search data and update validation info
        if company_name in self.search_data:
            for q, results in self.search_data[company_name].items():
                for result in results:
                    if result.get('url') == url:
                        # Update result with validation information
                        result['validation'] = validation_result
                        # Use the query from this result
                        query = q
                        break
        
        # Add entry to detailed data tracking
        self.add_data_entry(
            company_name=company_name,
            operation="Content Validation",
            status=f"{status} ({confidence})",
            details=details,
            url=url,
            content_preview=content_preview,
            validation_result=f"{status} - {reason}" if reason else status,
            search_query=query
        )
        
        # Also add to the company text log
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self.company_progress[company_name]["text"]:
            if is_valid:
                st.markdown(f"**{timestamp}**: Content validated ✓ ({confidence})")
            else:
                st.markdown(f"**{timestamp}**: Content invalid ✗ - {content_type}: {reason} ({confidence})")
    
    def log_extraction(self, company_name, field_name, extracted_value, success=True, error=None, source_url=""):
        """Enhanced method to log field extraction with source tracking"""
        status = "Success" if success else "Error"
        details = f"Extracted {field_name}" if success else f"Error extracting {field_name}: {error}"
        
        # Prepare preview of the extracted value
        value_preview = ""
        if extracted_value:
            if isinstance(extracted_value, dict):
                # For dictionaries, show first few key-value pairs
                value_preview = "; ".join([f"{k}: {v}" for k, v in list(extracted_value.items())[:3]])
                if len(extracted_value) > 3:
                    value_preview += f"... ({len(extracted_value) - 3} more)"
            elif isinstance(extracted_value, list):
                # For lists, show first few items
                if extracted_value:
                    value_preview = "; ".join([str(item) for item in extracted_value[:3]])
                    if len(extracted_value) > 3:
                        value_preview += f"... ({len(extracted_value) - 3} more)"
                else:
                    value_preview = "Empty list"
            else:
                # For other types, convert to string
                value_preview = str(extracted_value)
        
        # Find query associated with this URL
        query = ""
        if company_name in self.search_data and source_url:
            for q, results in self.search_data[company_name].items():
                for result in results:
                    if result.get('url') == source_url:
                        query = q
                        break
        
        # Add entry to detailed data tracking
        self.add_data_entry(
            company_name=company_name,
            operation=f"Extract {field_name}",
            status=status,
            details=details,
            content_preview=value_preview[:100] + "..." if value_preview and len(value_preview) > 100 else value_preview,
            url=source_url,
            search_query=query
        )
        
        # Also add to the company text log
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self.company_progress[company_name]["text"]:
            if success:
                st.markdown(f"**{timestamp}**: Extracted {field_name}: {value_preview[:50] + '...' if value_preview and len(value_preview) > 50 else value_preview}")
            else:
                st.markdown(f"**{timestamp}**: Failed to extract {field_name}: {error}")
    
    def log_entity_verification(self, company_name, entity_name, verification_result):
        """Log entity verification result with enhanced tracking"""
        # Extract verification information
        are_same_entity = verification_result.get('are_same_entity', False)
        confidence = verification_result.get('confidence_level', 'Unknown')
        relationship_type = verification_result.get('relationship_type', 'None')
        
        status = "Same Entity" if are_same_entity else f"Different Entity ({relationship_type})"
        
        # Get matching and differentiating attributes
        matching = verification_result.get('matching_attributes', [])
        differentiating = verification_result.get('differentiating_attributes', [])
        
        matching_str = ", ".join(matching[:3]) + (f"... ({len(matching) - 3} more)" if len(matching) > 3 else "")
        diff_str = ", ".join(differentiating[:3]) + (f"... ({len(differentiating) - 3} more)" if len(differentiating) > 3 else "")
        
        details = f"Target: {company_name}, Comparison: {entity_name}"
        validation_str = f"{status} ({confidence})\nMatching: {matching_str}\nDifferentiating: {diff_str}"
        
        # Add entry to detailed data tracking
        self.add_data_entry(
            company_name=company_name,
            operation="Entity Verification",
            status=f"{status} ({confidence})",
            details=details,
            validation_result=validation_str
        )
        
        # Also add to the company text log
        timestamp = datetime.now().strftime("%H:%M:%S")
        icon = "✅" if are_same_entity else "❌"
        with self.company_progress[company_name]["text"]:
            st.markdown(f"**{timestamp}**: Verified {entity_name}: {icon} {status} ({confidence})")
    
    def complete_company(self, company_name=None, success=True):
        """Marks a company as completed in the tracker."""
        if not company_name and self.current_company_idx < len(self.companies):
            company_name = self.companies[self.current_company_idx]["name"]
        
        # Update company status
        if company_name in self.company_progress:
            status_icon = "✅" if success else "❌"
            status_text = "Completed successfully" if success else "Processing failed"
            
            self.company_progress[company_name]["status"] = "Completed" if success else "Error"
            self.company_progress[company_name]["progress"] = 1.0
            self.company_progress[company_name]["bar"].progress(1.0)
            
            # Update within the expander
            with self.company_progress[company_name]["container"]:
                st.markdown(f"**Status**: {status_icon} {status_text}")
            
            # Add final timestamp with completion status
            timestamp = datetime.now().strftime("%H:%M:%S")
            with self.company_progress[company_name]["text"]:
                st.markdown(f"**{timestamp}**: {status_icon} {status_text}")
            
            # Add comprehensive completion summary
            if company_name in self.search_data:
                total_queries = len(self.search_data[company_name])
                total_results = sum(len(results) for results in self.search_data[company_name].values())
                
                with self.company_progress[company_name]["text"]:
                    st.markdown(f"**Search Summary**: Executed {total_queries} queries with {total_results} total results")
            
            # Add entry to detailed data tracking
            self.add_data_entry(
                company_name=company_name,
                operation="Complete Processing",
                status="Success" if success else "Failed",
                details=f"Company processing {status_text}"
            )
    
    def complete_processing(self):
        """Marks the entire batch processing as complete with comprehensive summary."""
        total_time = time.time() - self.start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)
        
        # Update overall status
        completion_message = f"Batch processing complete! Processed {self.total_companies} companies in {int(hours)}h {int(minutes)}m {int(seconds)}s"
        self.overall_status.success(completion_message)
        self.progress_bar.progress(1.0)
        self.eta.empty()
        
        # Compile comprehensive search statistics
        total_queries = sum(len(queries) for queries in self.search_data.values())
        total_results = sum(sum(len(results) for results in company_queries.values()) for company_queries in self.search_data.values())
        queries_per_company = total_queries / self.total_companies if self.total_companies > 0 else 0
        
        # Add final entry to detailed data tracking - WITH ERROR HANDLING
        try:
            self.add_data_entry(
                company_name="BATCH COMPLETE",
                operation="Batch Processing Complete",
                status="Success",
                details=completion_message,
                duration=total_time
            )
        except Exception as e:
            logger.error(f"Error adding completion entry: {e}")
        
        # Add a comprehensive summary to the dataframe display
        with self.dataframe_container:
            st.success(completion_message)
            
            # Display comprehensive search statistics
            st.subheader("Search Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Companies", self.total_companies)
            with col2:
                st.metric("Total Queries", total_queries)
            with col3:
                st.metric("Total Search Results", total_results)
            with col4:
                st.metric("Queries/Company", f"{queries_per_company:.1f}")
            
            # Create summary statistics
            operations = self.current_dataframe["Operation"].value_counts()
            statuses = self.current_dataframe["Status"].apply(lambda x: x.split(" ")[0]).value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Operations Summary")
                st.dataframe(operations.reset_index().rename(columns={"index": "Operation", "Operation": "Count"}))
            
            with col2:
                st.subheader("Status Summary")
                st.dataframe(statuses.reset_index().rename(columns={"index": "Status", 0: "Count"}))
            
            # Add download options for results
            if total_results > 0:
                st.subheader("Download Options")
                
                # Export search data as JSON
                search_data_json = json.dumps({
                    company: {
                        query: [
                            {k: v for k, v in result.items() if k not in ['raw_content']}  # Exclude raw_content to reduce size
                            for result in results
                        ]
                        for query, results in queries.items()
                    }
                    for company, queries in self.search_data.items()
                }, indent=2)
                
                st.download_button(
                    label="Download Complete Search Data (JSON)",
                    data=search_data_json,
                    file_name="search_results_data.json",
                    mime="application/json"
                )
                
                # Export operations dataframe as CSV
                csv_data = self.current_dataframe.to_csv(index=False)
                st.download_button(
                    label="Download Operations Log (CSV)",
                    data=csv_data,
                    file_name="operations_log.csv",
                    mime="text/csv"
                )
    
    def _update_eta(self):
        """Updates the estimated time remaining based on step timings."""
        if not self.step_times:
            return
        
        # Calculate average time per step
        avg_step_time = sum(sum(times) for times in self.step_times.values()) / sum(len(times) for times in self.step_times.values())
        
        # Estimate remaining time
        remaining_steps = self.total_steps - self.completed_steps
        estimated_remaining_seconds = remaining_steps * avg_step_time
        
        # Format time
        if estimated_remaining_seconds < 60:
            eta_text = f"{int(estimated_remaining_seconds)} seconds"
        elif estimated_remaining_seconds < 3600:
            minutes = estimated_remaining_seconds / 60
            eta_text = f"{int(minutes)} minutes"
        else:
            hours = estimated_remaining_seconds / 3600
            eta_text = f"{hours:.1f} hours"
        
        # Calculate percent complete
        percent_complete = (self.completed_steps / self.total_steps) * 100
        
        self.eta.info(f"**{percent_complete:.1f}%** complete. Estimated time remaining: **{eta_text}**")

class RowReviewResult(BaseModel):
    """Result of reviewing a single row, with details per field."""
    fields_to_review: List[str] = Field(
        default_factory=list, description="List of field names that need additional review and search."
    )
    needs_additional_search: bool = Field(False, description="True if additional web search is recommended for this row.")

class SearchQuerySuggestionResponse(BaseModel):
    """Response for suggesting a search query."""
    search_query: Optional[str] = Field(None, description="The suggested search query.")

class SearchQuerySet(BaseModel):
    """Set of specialized search queries for investment entity research."""
    general_queries: List[str] = Field(..., description="General search queries about the company")
    news_queries: List[str] = Field(..., description="Queries focused on recent news/developments")
    funding_queries: List[str] = Field(..., description="Queries specifically for funding information")
    relationship_queries: List[str] = Field(..., description="Queries for corporate relationships")

# Create the agent
investment_query_generation_agent = Agent(
    model=gemini_2o_model,
    result_type=SearchQuerySet,
    system_prompt="""You generate search queries for investment banking research.
    
For a given company/entity name, generate:
1. General queries that target fundamental entity information, funding, structure, and investors
2. News-specific queries that target recent developments (within last 2 years)

Focus on investment banking aspects such as:
- Funding rounds and amounts
- Investor relationships
- Valuation metrics
- Corporate structure
- M&A activity
- Financial performance

Queries should be precise and focused on investment banking research.
"""
)


async def generate_investment_search_queries_v4(company_name: str) -> SearchQuerySet:
    """
    Generate search queries for investment banking research.
    
    Args:
        company_name: Name of the company to search for
        
    Returns:
        SearchQuerySet with categorized search queries
    """
    try:
        query_result = await investment_query_generation_agent.run(
            user_prompt=f"Generate search queries for investment banking research on '{company_name}'."
        )
        
        return query_result.data
    except Exception as e:
        logger.error(f"Error generating investment search queries: {e}")
        # Return a default set of queries
        return SearchQuerySet(
            general_queries=[
                f"{company_name} company information",
                f"{company_name} business overview"
            ],
            news_queries=[
                f"{company_name} recent news",
                f"{company_name} latest developments"
            ],
            funding_queries=[
                f"{company_name} funding rounds",
                f"{company_name} investors"
            ],
            relationship_queries=[
                f"{company_name} acquisitions partnerships",
                f"{company_name} parent company subsidiaries"
            ]
        )


# Entity Database and Double-Check Workflow

# 1. Pydantic Models
class FlatEntityAttributes(BaseModel):
    """Flat structure of entity attributes for Gemini agent compatibility"""
    entity_name: str = Field(..., description="Name of the entity as mentioned in text")
    similar_names: List[str] = Field(default_factory=list, description="Similar or alternative names mentioned")
    
    # Founder information
    founder_names: List[str] = Field(default_factory=list, description="Names of founders")
    founder_mentions: List[str] = Field(default_factory=list, description="Raw text mentions of founders")
    
    # Temporal information
    founding_year: Optional[int] = Field(None, description="Year the entity was founded")
    acquisition_year: Optional[int] = Field(None, description="Year the entity was acquired, if applicable")
    rebranding_year: Optional[int] = Field(None, description="Year the entity was rebranded, if applicable")
    latest_known_activity: Optional[str] = Field(None, description="Latest known activity or mention with date")
    
    # Location information
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    additional_locations: List[str] = Field(default_factory=list, description="Additional office locations")
    
    # Funding information
    total_funding_amount: Optional[str] = Field(None, description="Total funding amount (textual representation)")
    latest_funding_round: Optional[str] = Field(None, description="Latest funding round information")
    investors: List[str] = Field(default_factory=list, description="Known investors")
    
    # Product information
    main_products: List[str] = Field(default_factory=list, description="Main products or services")
    industry_focus: List[str] = Field(default_factory=list, description="Industry focus areas")
    technologies: List[str] = Field(default_factory=list, description="Key technologies used or developed")
    
    # Relationship information
    parent_company: Optional[str] = Field(None, description="Parent company if subsidiary")
    subsidiaries: List[str] = Field(default_factory=list, description="Known subsidiaries")
    previous_names: List[str] = Field(default_factory=list, description="Previous company names")
    related_entities: List[str] = Field(default_factory=list, description="Other related entities and their relationship")

class EntityProfile(BaseModel):
    """Profile of an entity stored in the database."""
    entity_name: str = Field(..., description="Primary name of the entity")
    alternative_names: List[str] = Field(default_factory=list, description="Alternative names for the same entity")
    attributes: FlatEntityAttributes = Field(..., description="Extracted attributes of the entity")
    related_entities: Dict[str, str] = Field(default_factory=dict, description="Related entities and their relationship types")
    last_updated: str = Field(..., description="Timestamp when this profile was last updated")
    sources: List[str] = Field(default_factory=list, description="Sources used to build this profile")

class EntityVerificationResult(BaseModel):
    """Result of verifying if two entities are the same."""
    are_same_entity: bool = Field(..., description="Whether the entities are the same")
    primary_entity: str = Field(..., description="Name of the primary entity being searched")
    comparison_entity: str = Field(..., description="Name of the entity being compared")
    matching_attributes: List[str] = Field(default_factory=list, description="Attributes that match between entities")
    differentiating_attributes: List[str] = Field(default_factory=list, description="Attributes that differentiate the entities")
    relationship_type: Optional[str] = Field(None, description="Type of relationship if not the same entity")
    confidence_level: str = Field(..., description="Confidence level: Low, Medium, High, Very High")
    
# 2. Entity Database Class
class EntityDatabase:
    """Database for storing and retrieving entity profiles."""
    
    def __init__(self):
        """Initialize an empty entity database."""
        self.entities = {}  # name -> EntityProfile
        self.name_mapping = {}  # alternative name -> primary name
    
    def add_entity(self, profile: EntityProfile):
        """Add or update an entity profile in the database."""
        self.entities[profile.entity_name] = profile
        
        # Update name mapping for alternative names
        for alt_name in profile.alternative_names:
            self.name_mapping[alt_name.lower()] = profile.entity_name
    
    def get_entity(self, name: str) -> Optional[EntityProfile]:
        """Get an entity profile by name (primary or alternative)."""
        # Check if it's a primary name
        if name in self.entities:
            return self.entities[name]
        
        # Check if it's an alternative name
        if name.lower() in self.name_mapping:
            primary_name = self.name_mapping[name.lower()]
            return self.entities[primary_name]
        
        return None
    
    def are_related_entities(self, name1: str, name2: str) -> Tuple[bool, Optional[str]]:
        """Check if two entities are related."""
        entity1 = self.get_entity(name1)
        entity2 = self.get_entity(name2)
        
        if not entity1 or not entity2:
            return False, None
        
        # Check if they are the same entity
        if entity1.entity_name == entity2.entity_name:
            return True, "same_entity"
        
        # Check if one is in the related entities of the other
        if entity2.entity_name in entity1.related_entities:
            return True, entity1.related_entities[entity2.entity_name]
        
        if entity1.entity_name in entity2.related_entities:
            return True, entity2.related_entities[entity1.entity_name]
        
        return False, None
    
    def mark_as_related(self, name1: str, name2: str, relationship_type: str):
        """Mark two entities as related."""
        entity1 = self.get_entity(name1)
        entity2 = self.get_entity(name2)
        
        if entity1 and entity2:
            entity1.related_entities[entity2.entity_name] = relationship_type
            entity2.related_entities[entity1.entity_name] = relationship_type
    
    def add_alternative_name(self, primary_name: str, alt_name: str):
        """Add an alternative name for an entity."""
        if primary_name in self.entities:
            profile = self.entities[primary_name]
            if alt_name not in profile.alternative_names:
                profile.alternative_names.append(alt_name)
                self.name_mapping[alt_name.lower()] = primary_name

# 3. Entity Verification Agent
entity_verification_agent = Agent(
    model=gemini_2o_model,
    result_type=EntityVerificationResult,
    system_prompt="""You are an expert at verifying if two entities are the same or different businesses.

Your task is to analyze two entity profiles and determine if they represent the same business entity or different ones.

Follow this systematic analysis approach:

1. Compare core identity attributes:
   - Company names and name patterns
   - Founding information (year founded, founders)
   - Headquarters location
   - Business focus and industry

2. Look for clear differentiators:
   - Different founding years (especially if >1 year apart)
   - Different founders or founding teams
   - Incompatible business descriptions
   - Different product/service offerings

3. Identify potential relationships if they're different entities:
   - Subsidiary relationship
   - Parent company relationship
   - Rebranding/rename
   - Merger/acquisition
   - Spin-off

4. Determine your level of confidence based on:
   - Quantity of matching/differentiating attributes
   - Quality and reliability of the information
   - Presence of definitive differentiators
   - Explicit statements about relationships

Return your analysis as an EntityVerificationResult with:
- Whether the entities are the same
- The names of both entities
- Lists of matching and differentiating attributes
- Relationship type if not the same entity
- Your confidence level (Low, Medium, High, Very High)
"""
)

# 4. Main Verification Functions
async def double_check_potentially_relevant_entities(
    target_company_name: str,
    potentially_relevant_entities: List[Dict],
    entity_database: EntityDatabase,
    search_results_all_companies: Dict,
    tracker: Optional[BatchProcessTracker] = None
) -> Dict[str, EntityVerificationResult]:
    """Double-check potentially relevant entities against the entity database."""
    # Get target company profile
    target_company_data = search_results_all_companies.get(target_company_name, {})
    target_features = target_company_data.get('company_features')
    
    # Make sure we have the target company in the database
    if not entity_database.get_entity(target_company_name):
        # Extract target company attributes from search results
        # FIX: Added safety checks for None values with or "" to prevent TypeError
        target_content = " ".join([
            result.get('search_result_metadata', {}).get('content', "") or ""  # Add or "" to handle None
            for result in target_company_data.get('results_with_metadata', []) 
            if (result.get('extracted_company_name', '') and target_company_name and 
                result.get('extracted_company_name', '').lower() == target_company_name.lower())
        ]) or ""
        
        # Extract entity attributes
        target_attributes = await extract_entity_attributes(
            content_text=target_content,
            entity_name=target_company_name
        )
        
        # Create and add profile to database
        target_profile = EntityProfile(
            entity_name=target_company_name,
            alternative_names=[],
            attributes=target_attributes,
            last_updated=datetime.now().isoformat(),
            sources=[result.get('search_result_metadata', {}).get('url', "") or ""  # Add or "" for safety
                    for result in target_company_data.get('results_with_metadata', [])]
        )
        entity_database.add_entity(target_profile)
    
    # Extract unique entity names from potentially relevant results
    unique_entity_names = set()
    entity_to_result_mapping = {}  # Map entity names to their search results
    
    for result in potentially_relevant_entities:
        entity_name = result.get('extracted_company_name')
        if entity_name and entity_name != target_company_name:
            unique_entity_names.add(entity_name)
            if entity_name not in entity_to_result_mapping:
                entity_to_result_mapping[entity_name] = []
            entity_to_result_mapping[entity_name].append(result)
    
    # Initialize verification results
    verification_results = {}
    target_profile = entity_database.get_entity(target_company_name)
    
    # Check and search for each unique entity
    verification_tasks = []
    entity_search_tasks = []
    entity_names_to_search = []
    
    # First check which entities we need to search for
    for entity_name in unique_entity_names:
        # Check if already in database
        if entity_database.get_entity(entity_name):
            # If already in database, create verification task
            verification_tasks.append(
                verify_entity_relationship(
                    target_profile=target_profile,
                    comparison_entity_name=entity_name,
                    comparison_profile=entity_database.get_entity(entity_name),
                    entity_database=entity_database
                )
            )
        else:
            # Need to search for this entity
            entity_names_to_search.append(entity_name)
            entity_search_tasks.append(
                search_entity_information(entity_name)
            )
    
    # Execute search tasks in parallel
    if entity_search_tasks:
        entity_search_results = await asyncio.gather(*entity_search_tasks)
        
        # Process search results and create entity profiles
        for i, entity_name in enumerate(entity_names_to_search):
            search_results = entity_search_results[i]
            
            # Process search results to extract entity attributes
            # FIX: Added safety checks for None values
            entity_content = " ".join([
                result.get('content', "") or ""  # Add or "" to handle None
                for result in search_results
            ]) or ""
            
            # Extract entity attributes
            entity_attributes = await extract_entity_attributes(
                content_text=entity_content,
                entity_name=entity_name
            )
            
            # Create and add profile to database
            entity_profile = EntityProfile(
                entity_name=entity_name,
                alternative_names=[],
                attributes=entity_attributes,
                last_updated=datetime.now().isoformat(),
                sources=[result.get('url', "") or "" for result in search_results]  # Add or "" for safety
            )
            entity_database.add_entity(entity_profile)
            
            # Create verification task for this entity
            verification_tasks.append(
                verify_entity_relationship(
                    target_profile=target_profile,
                    comparison_entity_name=entity_name,
                    comparison_profile=entity_profile,
                    entity_database=entity_database
                )
            )
    
    # Execute all verification tasks in parallel
    if verification_tasks:
        verification_results_list = await asyncio.gather(*verification_tasks)
        
        # Process verification results
        for result in verification_results_list:
            if result:
                verification_results[result.comparison_entity] = result
                
                # Log verification results if tracker provided
                if tracker:
                    tracker.log_entity_verification(
                        company_name=target_company_name,
                        entity_name=result.comparison_entity,
                        verification_result=result.model_dump()
                    )
                
                # Update database with relationship if different entities
                if not result.are_same_entity and result.relationship_type:
                    entity_database.mark_as_related(
                        name1=result.primary_entity,
                        name2=result.comparison_entity,
                        relationship_type=result.relationship_type
                    )
                # Update alternative names if same entity
                elif result.are_same_entity:
                    entity_database.add_alternative_name(
                        primary_name=result.primary_entity,
                        alt_name=result.comparison_entity
                    )
    
    return verification_results

async def verify_entity_relationship(
    target_profile: EntityProfile,
    comparison_entity_name: str,
    comparison_profile: EntityProfile,
    entity_database: EntityDatabase,
    tracker: Optional[BatchProcessTracker] = None,
    company_name: Optional[str] = None
) -> Optional[EntityVerificationResult]:
    """Verify the relationship between two entities using the entity verification agent."""
    try:
        # Check if we already know the relationship
        are_related, relationship_type = entity_database.are_related_entities(
            target_profile.entity_name, comparison_entity_name
        )
        
        if are_related:
            # If already known to be the same entity
            if relationship_type == "same_entity":
                return EntityVerificationResult(
                    are_same_entity=True,
                    primary_entity=target_profile.entity_name,
                    comparison_entity=comparison_entity_name,
                    matching_attributes=["Previously verified as same entity"],
                    differentiating_attributes=[],
                    relationship_type=None,
                    confidence_level="Very High"
                )
            # If already known to have a specific relationship
            elif relationship_type:
                return EntityVerificationResult(
                    are_same_entity=False,
                    primary_entity=target_profile.entity_name,
                    comparison_entity=comparison_entity_name,
                    matching_attributes=[],
                    differentiating_attributes=["Previously identified as different entities"],
                    relationship_type=relationship_type,
                    confidence_level="Very High"
                )
        
        # Format entity data for verification
        target_data = {
            "name": target_profile.entity_name,
            "attributes": target_profile.attributes.model_dump()
        }
        
        comparison_data = {
            "name": comparison_profile.entity_name,
            "attributes": comparison_profile.attributes.model_dump()
        }
        
        # Run verification agent with improved error handling
        try:
            async with sem:  # Use semaphore for rate limiting
                verification_result = await entity_verification_agent.run(
                    user_prompt=f"""
                    Compare these two entity profiles to determine if they represent the same business entity:
                    
                    TARGET ENTITY:
                    {json.dumps(target_data, indent=2)}
                    
                    COMPARISON ENTITY:
                    {json.dumps(comparison_data, indent=2)}
                    
                    Provide your determination on whether these are the same entity or different ones.
                    """
                )
            
            # Log entity verification if tracker provided
            if tracker and company_name and verification_result and verification_result.data:
                tracker.log_entity_verification(
                    company_name=company_name,
                    entity_name=comparison_entity_name,
                    verification_result=verification_result.data.model_dump()
                )
            
            return verification_result.data
            
        except RuntimeError as e:
            # Handle event loop errors
            if "different event loop" in str(e):
                logger.warning(f"Event loop issue in verification. Creating fallback result for {comparison_entity_name}")
                # Create a fallback result with low confidence
                return EntityVerificationResult(
                    are_same_entity=False,  # Default to not same entity for safety
                    primary_entity=target_profile.entity_name,
                    comparison_entity=comparison_entity_name,
                    matching_attributes=[],
                    differentiating_attributes=["Verification skipped due to technical limitations"],
                    relationship_type="unknown",
                    confidence_level="Low" 
                )
            else:
                raise  # Re-raise other RuntimeErrors
                
    except Exception as e:
        logger.error(f"Error verifying entity relationship: {str(e)}")
        # Return a default result for error case
        return EntityVerificationResult(
            are_same_entity=False,
            primary_entity=target_profile.entity_name,
            comparison_entity=comparison_entity_name,
            matching_attributes=[],
            differentiating_attributes=[f"Error during verification: {str(e)}"],
            relationship_type="unknown",
            confidence_level="Low"
        )

async def search_entity_information(entity_name: str) -> List[Dict]:
    """Search for information about an entity using Tavily."""
    try:
        # Generate search query
        search_query = f"{entity_name} company information business"
        
        # Execute search with proper error handling
        async with sem:
            search_results_raw = await tavily_client.search(
                query=search_query,
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True
            )
            
        if not search_results_raw or not search_results_raw.get('results'):
            logger.warning(f"No results found for entity: {entity_name}")
            return []
            
        return search_results_raw.get('results', [])
        
    except Exception as e:
        logger.error(f"Error searching for entity '{entity_name}': {e}")
        return []

def initialize_entity_database():
    """Initialize the entity database in session state."""
    if 'entity_database' not in st.session_state:
        st.session_state.entity_database = EntityDatabase()

def handle_tavily_credit_exhaustion(exception, company_name=None, tracker=None):
    """
    Handle Tavily API credit exhaustion errors (403 Forbidden).
    
    Args:
        exception: The exception that occurred during the API call
        company_name: Optional company name for tracking purposes
        tracker: Optional BatchProcessTracker for logging the error
        
    Returns:
        bool: True if credits are exhausted, False otherwise
    """
    # Check if this is a credit exhaustion error (403 Forbidden)
    is_credit_exhausted = False
    
    if (hasattr(exception, 'response') and 
        getattr(exception.response, 'status_code', None) == 403):
        is_credit_exhausted = True
        
        # Create a prominent error message for the user
        st.error("⚠️ Tavily API credits exhausted")
        st.warning(
            "Your Tavily API credits have been depleted. Please add more credits "
            "to your Tavily AI account to continue using the search functionality."
        )
        
        # Provide guidance on how to add more credits
        st.info(
            "To add more credits to your Tavily account:\n"
            "1. Visit [Tavily Pricing Page](https://tavily.com/pricing)\n"
            "2. Sign in to your account\n"
            "3. Select an appropriate plan or add more credits to your existing plan"
        )
        
        # Log the error if tracker is provided
        if tracker and company_name:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Tavily API",
                status="Credits Exhausted",
                details="Tavily API credits have been exhausted. Search functionality is limited."
            )
    
    return is_credit_exhausted

@st.fragment
async def check_tavily_api_status():
    """
    Check Tavily API status and credit availability.
    
    Returns:
        Tuple of (is_available, has_credits, message)
    """
    try:
        # Make a minimal API call to verify status
        result = await tavily_client.search(
            query="api status check",
            max_results=1,
            search_depth="basic"
        )
        return True, True, "Tavily API is available and has credits."
    except Exception as e:
        if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 403:
            return True, False, "Tavily API credits exhausted."
        else:
            return False, False, f"Tavily API connection issue: {str(e)}"

# 5. Modified Multi-Source Search Function
async def execute_multi_source_search_v4(company_name: str, use_cache: bool = True, force_refresh: bool = False, tracker=None) -> List[Dict]:
    """
    Enhanced V4 version of multi-source search with entity database filtering, search tracking,
    and cache integration.
    """
    try:
        # Use provided company name for tracking if available
        tracking_company_name = company_name
        
        # Check for known non-matching entities in the database
        non_matching_entities = []
        if 'entity_database' in st.session_state:
            entity_db = st.session_state.entity_database
            target_entity = entity_db.get_entity(company_name)
            
            if target_entity and target_entity.related_entities:
                # Get all related entities that are verified as different
                for related_name, relationship in target_entity.related_entities.items():
                    if relationship != "same_entity":
                        non_matching_entities.append(related_name)
        
        # Generate specialized queries
        search_queries = await generate_investment_search_queries_v4(company_name)
        
        # Create search tasks with explicit query tracking
        search_tasks = []
        query_mapping = {}  # Keep track of which query is used for which task
        
        # Execute all query types (general, news, funding, relationship)
        for query_type in ['general_queries', 'news_queries', 'funding_queries', 'relationship_queries']:
            for query in getattr(search_queries, query_type, []):
                max_results = 7 if query_type == 'funding_queries' else 5
                
                # Map query type to the appropriate search function and cache type
                search_function = None
                cache_type = None
                
                if query_type == 'general_queries' or query_type == 'relationship_queries':
                    search_function = execute_general_search
                    cache_type = "general"
                elif query_type == 'news_queries':
                    search_function = execute_news_search
                    cache_type = "news"
                elif query_type == 'funding_queries':
                    search_function = execute_funding_search
                    cache_type = "funding"
                
                if search_function:
                    # Use tracker-enabled search function if tracker provided
                    if tracker:
                        task = execute_search_with_query_tracking(
                            query, 
                            max_results=max_results,
                            tracker=tracker,
                            company_name=tracking_company_name,
                            use_cache=use_cache,
                            force_refresh=force_refresh,
                            cache_type=cache_type
                        )
                    else:
                        task = search_function(
                            query, 
                            max_results=max_results, 
                            use_cache=use_cache, 
                            force_refresh=force_refresh
                        )
                        
                    search_tasks.append(task)
                    query_mapping[id(task)] = query
        
        # Execute all search tasks concurrently
        all_results_nested = await asyncio.gather(*search_tasks)
        
        # Process results the same as before
        all_results = []
        seen_urls = set()
        
        for result_list in all_results_nested:
            for result in result_list:
                url = result.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    
                    # Ensure search_query is present
                    if 'search_query' not in result:
                        for task_id, query in query_mapping.items():
                            if result in result_list:
                                result['search_query'] = query
                                break
                    
                    # Check if this result mentions any known non-matching entities
                    content = result.get('content', '').lower()
                    mentions_non_matching = False
                    
                    for non_matching_entity in non_matching_entities:
                        if non_matching_entity.lower() in content:
                            # Check if it mentions the non-matching entity more prominently than target
                            target_count = content.count(company_name.lower())
                            non_match_count = content.count(non_matching_entity.lower())
                            
                            if non_match_count > target_count:
                                logger.info(f"Filtering result mentioning non-matching entity: {non_matching_entity}")
                                # Track filtering if tracker provided
                                if tracker:
                                    tracker.add_data_entry(
                                        company_name=tracking_company_name,
                                        operation="Filter Result",
                                        status="Non-matching Entity",
                                        details=f"Filtered result mentioning non-matching entity: {non_matching_entity}",
                                        url=url,
                                        content_preview=content[:100]
                                    )
                                mentions_non_matching = True
                                break
                    
                    # Only add results that don't prominently mention non-matching entities
                    if not mentions_non_matching:
                        all_results.append(result)
        
        # Track final results count if tracker provided
        if tracker:
            tracker.add_data_entry(
                company_name=tracking_company_name,
                operation="Multi-Source Search",
                status="Completed",
                details=f"Found {len(all_results)} unique results after filtering",
                result_count=len(all_results)
            )
            
        return all_results
    except Exception as e:
        logger.error(f"Error in v4 multi-source search: {e}")
        # Track error if tracker provided
        if tracker:
            tracker.add_data_entry(
                company_name=tracking_company_name,
                operation="Multi-Source Search",
                status="Error",
                details=f"Error in search: {str(e)}"
            )
        # Fall back to original function with cache settings
        return await execute_multi_source_search(company_name, use_cache, force_refresh)


def fix_missing_search_queries(results_with_metadata):
    """
    Attempts to fix missing search queries in existing results.
    
    Args:
        results_with_metadata: List of search results with metadata
        
    Returns:
        Updated list with search queries where possible
    """
    if not results_with_metadata:
        return results_with_metadata
    
    # Common search query patterns to use as fallbacks
    default_queries = [
        "{company_name} company information",
        "{company_name} funding investors",
        "{company_name} founders team",
        "{company_name} product technology",
        "{company_name} location headquarters"
    ]
    
    # Extract company name from first result if possible
    company_name = ""
    for result in results_with_metadata:
        if result.get('extracted_company_name'):
            company_name = result['extracted_company_name']
            break
    
    # Apply company name to default queries
    default_queries = [q.format(company_name=company_name) for q in default_queries]
    
    # Count results with missing queries
    missing_count = 0
    fixed_count = 0
    
    for i, result in enumerate(results_with_metadata):
        # Check if search_query is missing
        has_query = False
        
        if result.get('search_result_metadata', {}).get('search_query'):
            has_query = True
        elif result.get('search_query'):
            # Move it to the standardized location
            if 'search_result_metadata' not in result:
                result['search_result_metadata'] = {}
            result['search_result_metadata']['search_query'] = result['search_query']
            has_query = True
            fixed_count += 1
        
        if not has_query:
            missing_count += 1
            # Assign a default query based on result index
            default_idx = i % len(default_queries)
            
            if 'search_result_metadata' not in result:
                result['search_result_metadata'] = {}
            
            result['search_result_metadata']['search_query'] = default_queries[default_idx]
            result['search_query'] = default_queries[default_idx]
            fixed_count += 1
    
    st.info(f"Fixed {fixed_count} results with missing search queries. {missing_count} results had missing queries.")
    return results_with_metadata


async def enhanced_search_company_summary_v4(
    company_name: str, 
    company_urls: Optional[List[str]] = None, 
    use_test_data: bool = False, 
    NUMBER_OF_SEARCH_RESULTS: int = 5, 
    original_input: Optional[str] = None,
    # Added parameters for tracker integration
    tracker: Optional[BatchProcessTracker] = None,
    query_container: Optional[Any] = None,
    search_results_container: Optional[Any] = None,
    company_data_container: Optional[Any] = None,
    # New parameters for cache control
    use_cache: bool = True,
    force_refresh: bool = False
) -> tuple:
    """
    Enhanced V4 version with entity database integration, search visualization,
    and integrated cache control.
    """
    try:
        # Use cache settings from session state if not explicitly provided
        if 'use_cache' in st.session_state and use_cache is True:
            use_cache = st.session_state.use_cache
            
        if 'force_refresh' in st.session_state and force_refresh is False:
            force_refresh = st.session_state.force_refresh
        
        # Track cache usage if tracker provided
        if tracker:
            cache_status = "enabled" if use_cache else "disabled"
            refresh_status = "with forced refresh" if force_refresh else "using cache when available"
            tracker.add_data_entry(
                company_name=company_name,
                operation="Cache Settings",
                status="Info",
                details=f"Search cache {cache_status} {refresh_status}"
            )
        
        # Check entity database for known relationships (unchanged)
        known_entity_relationships = []
        if 'entity_database' in st.session_state:
            entity_db = st.session_state.entity_database
            target_entity = entity_db.get_entity(company_name)
            
            if target_entity and target_entity.related_entities:
                # Get all entity relationships
                for related_name, relationship_type in target_entity.related_entities.items():
                    known_entity_relationships.append({
                        "entity_name": related_name,
                        "relationship": relationship_type
                    })
        
        # Use enhanced multi-source search with cache parameters
        all_search_results = await execute_multi_source_search_v4(
            company_name, 
            tracker=tracker,  # Pass tracker through
            use_cache=use_cache,  # Pass cache control parameter
            force_refresh=force_refresh  # Pass refresh control parameter
        )

        
        # Extract company name same as original function
        async with sem:
            company_name_extraction_output = await company_name_agent.run(
                user_prompt=f"Extract the company name from the following text: {company_name}"
            )
        cleaned_search_query_name = basename(company_name_extraction_output.data.company_name).lower() if company_name_extraction_output.data.company_name else basename(company_name).lower()
        
        # Process content same as original, but use all_search_results from enhanced search
        processing_tasks = []
        for result in all_search_results:
            # Track content processing if tracker is provided
            if tracker:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Process Content",
                    status="Started",
                    details=f"Processing content from {result.get('title', 'Untitled')}",
                    url=result.get('url', ''),
                    content_preview=result.get('content', '')[:100] if result.get('content') else '',
                    search_query=result.get('search_query', '')
                )
                
            processing_tasks.append(process_content(
                content_text=result.get('raw_content', result.get('content', '')),
                title=result.get('title', ''),
                url=result.get('url', ''),
                extracted_content_gold_standard=None,
                tracker=tracker,
                company_name=company_name
            ))

        
        # Use enhanced differentiation when processing results
        processed_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Organize results with metadata - WITH ADDED ERROR HANDLING
        results_with_metadata = []
        for index, processed_result in enumerate(processed_results):
            result_dict = {
                "search_result_metadata": {
                    "title": all_search_results[index].get('title', ''),
                    "url": all_search_results[index].get('url', ''),
                    "content": all_search_results[index].get('content', ''),
                    "raw_content": all_search_results[index].get('raw_content', ''),
                    "search_result_index": index,
                    "search_query": all_search_results[index].get('search_query', ''),
                    "v4_enhanced": True,
                    "source_type": all_search_results[index].get('source_type', 'general')
                }
            }
            
            if isinstance(processed_result, Exception):
                result_dict["error"] = str(processed_result)
                # Log error if tracker provided
                if tracker:
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation="Process Content",
                        status="Error",
                        details=f"Error processing content: {str(processed_result)}",
                        url=all_search_results[index].get('url', ''),
                        search_query=all_search_results[index].get('search_query', '')
                    )
            elif isinstance(processed_result, dict):
                result_dict.update(processed_result)
                
                # CRITICAL FIX: Add safe handling of content preview
                if tracker:
                    # Safely extract description_abstract if it exists
                    description = None
                    if processed_result.get('company_data') and isinstance(processed_result['company_data'], dict):
                        description = processed_result['company_data'].get('description_abstract', '')
                    
                    # Safely create preview with proper type checking
                    preview = ""
                    if description and isinstance(description, str):
                        preview = description[:100]
                    
                    # Log with safe content preview
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation="Process Content",
                        status="Success",
                        details=f"Extracted company name: {processed_result.get('extracted_company_name', 'None')}",
                        url=all_search_results[index].get('url', ''),
                        content_preview=preview,
                        search_query=all_search_results[index].get('search_query', '')
                    )
            else:
                result_dict["error"] = "Unexpected result type"
                # Log error if tracker provided
                if tracker:
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation="Process Content",
                        status="Error",
                        details="Unexpected result type",
                        url=all_search_results[index].get('url', ''),
                        search_query=all_search_results[index].get('search_query', '')
                    )
            
            results_with_metadata.append(result_dict)

        
        # Process results with known entity relationships
        if known_entity_relationships:
            for result in results_with_metadata:
                entity_name = result.get("extracted_company_name", "")
                
                # Check if this result is about a known related entity
                for relationship in known_entity_relationships:
                    if entity_name.lower() == relationship["entity_name"].lower():
                        # Mark this result with the relationship information
                        result["known_relationship"] = relationship["relationship"]
                        result["is_different_entity"] = (relationship["relationship"] != "same_entity")
                        
                        # Log the relationship for debugging
                        logger.info(f"Found result for known related entity: {entity_name} ({relationship['relationship']})")
                        
                        # Track relationship if tracker provided
                        if tracker:
                            tracker.add_data_entry(
                                company_name=company_name,
                                operation="Entity Relationship",
                                status="Found",
                                details=f"Found known relationship with {entity_name}: {relationship['relationship']}",
                                url=result.get('search_result_metadata', {}).get('url', '')
                            )
        
        # Separate exact and non-exact matches
        exact_match_results_metadata = []
        non_exact_match_results_metadata = []
        
        for result_dict in results_with_metadata:
            # Skip results that are known to be different entities
            if result_dict.get("is_different_entity"):
                logger.info(f"Skipping known different entity: {result_dict.get('extracted_company_name')}")
                # Track skipped entity if tracker provided
                if tracker:
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation="Skip Entity",
                        status="Different Entity",
                        details=f"Skipping known different entity: {result_dict.get('extracted_company_name')}",
                        url=result_dict.get('search_result_metadata', {}).get('url', '')
                    )
                continue
                
            if result_dict.get("extracted_company_name") and cleaned_search_query_name and result_dict.get("extracted_company_name").lower() == cleaned_search_query_name.lower():
                exact_match_results_metadata.append(result_dict)
                # Track exact match if tracker provided
                if tracker:
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation="Classify Result",
                        status="Exact Match",
                        details=f"Exact match found: {result_dict.get('extracted_company_name')}",
                        url=result_dict.get('search_result_metadata', {}).get('url', '')
                    )
            else:
                # Only add non-exact matches if not marked as different entity
                non_exact_match_results_metadata.append(result_dict)
                # Track non-exact match if tracker provided
                if tracker:
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation="Classify Result",
                        status="Non-Exact Match",
                        details=f"Non-exact match found: {result_dict.get('extracted_company_name')}",
                        url=result_dict.get('search_result_metadata', {}).get('url', '')
                    )
        
        # Use enhanced negative examples identification
        basic_company_info = ""
        if exact_match_results_metadata:
            for result in exact_match_results_metadata[:2]:
                if result.get('company_data'):
                    company_data = result.get('company_data')
                    if isinstance(company_data, dict):
                        basic_company_info += f"Industry: {company_data.get('product_type', 'Unknown')}\n"
                        basic_company_info += f"Description: {company_data.get('description_abstract', 'Unknown')}\n"
                        break
        
        # Track negative examples identification if tracker provided
        if tracker:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Identify Negative Examples",
                status="Started",
                details=f"Identifying negative examples from {len(non_exact_match_results_metadata)} non-exact matches"
            )
            
        negative_examples_collection = await enhanced_identify_negative_examples_v4(
            target_company_name=cleaned_search_query_name,
            basic_company_info=basic_company_info,
            search_results=non_exact_match_results_metadata
        )
        
        # Track negative examples results if tracker provided
        if tracker and negative_examples_collection and hasattr(negative_examples_collection, 'negative_examples'):
            tracker.add_data_entry(
                company_name=company_name,
                operation="Identify Negative Examples",
                status="Completed",
                details=f"Found {len(negative_examples_collection.negative_examples)} negative examples"
            )

        extracted_content_for_agent = result_dict.get('raw_content') if result_dict.get('raw_content') else None
        extracted_content_available_message = "Extracted content IS available." if extracted_content_for_agent else "Extracted content is NOT available."
        
        # Track selection agent processing if tracker provided
        if tracker:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Selection Agent",
                status="Started",
                details=f"Running selection agent on {len(non_exact_match_results_metadata)} non-exact matches"
            )
        
        # Run selection agent for non-exact matches
        selection_tasks = []
        for result_dict in non_exact_match_results_metadata:
            selection_tasks.append(
                gemini_selection_agent.run(
                    user_prompt=f"Search Query Company Name Cleaned: {cleaned_search_query_name}\nExact Match Results Metadata: {exact_match_results_metadata}\nExtracted Content: {extracted_content_for_agent}\nExtracted Content Available Message: {extracted_content_available_message}\nSearch Result: {result_dict}"
                )
            )
                                
        if selection_tasks:
            selection_results = await asyncio.gather(*selection_tasks)
            
            # Track selection results if tracker provided
            if tracker:
                included_count = sum(1 for result in selection_results if hasattr(result, 'data') and result.data and result.data.include_in_table)
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Selection Agent",
                    status="Completed",
                    details=f"Selected {included_count} out of {len(selection_results)} non-exact matches"
                )
        else:
            selection_results = []
            # Track empty selection if tracker provided
            if tracker:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Selection Agent",
                    status="Skipped",
                    details="No non-exact matches to process"
                )
        
        # Extract company features
        context_for_features = "\n\n".join([
            res['search_result_metadata']['content'] 
            for res in exact_match_results_metadata 
            if res.get('search_result_metadata') and res['search_result_metadata'].get('content')
        ])
        
        if not context_for_features and results_with_metadata:
            context_for_features = results_with_metadata[0]['search_result_metadata'].get('content', '') if results_with_metadata[0].get('search_result_metadata') else ""
        
        # Track feature extraction if tracker provided
        if tracker:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Extract Features",
                status="Started",
                details=f"Extracting company features from {len(exact_match_results_metadata)} exact matches"
            )
        
        if context_for_features:
            company_features_output = await company_features_agent.run(
                user_prompt=f"Company Name: {cleaned_search_query_name}\nExact Match Company Data List: {[res.get('company_data', {}) for res in exact_match_results_metadata]}\nContext Text:\n{context_for_features}"
            )
            company_features = company_features_output.data
            
            # Track successful feature extraction if tracker provided
            if tracker:
                feature_count = sum(1 for field, value in company_features.model_dump().items() if value)
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Extract Features",
                    status="Completed",
                    details=f"Extracted {feature_count} company features"
                )
        else:
            company_features = None
            # Track skipped feature extraction if tracker provided
            if tracker:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Extract Features",
                    status="Skipped",
                    details="Insufficient context for feature extraction"
                )
        
        # Group results by entity name (exact matches)
        grouped_results_dict = {}
        for result_dict in exact_match_results_metadata:
            entity_name_raw = result_dict.get("company_data", {}).get("company_name") or result_dict.get("extracted_company_name") or "Unknown Entity"
            entity_name_cleaned = basename(entity_name_raw) if entity_name_raw and entity_name_raw != "Unknown Entity" else "Unknown Entity"
            entity_name = entity_name_cleaned.lower() if entity_name_cleaned else "unknown"
            
            if entity_name not in grouped_results_dict:
                grouped_results_dict[entity_name] = []
            
            grouped_results_dict[entity_name].append(result_dict)
        
        # Final summary with cache information if tracker provided
        if tracker:
            cache_info = "cached results" if use_cache and not force_refresh else "fresh search results"
            tracker.add_data_entry(
                company_name=company_name,
                operation="Search Summary",
                status="Completed",
                details=f"Found {len(exact_match_results_metadata)} exact matches and {len(non_exact_match_results_metadata)} non-exact matches using {cache_info}. Identified {len(negative_examples_collection.negative_examples) if negative_examples_collection and hasattr(negative_examples_collection, 'negative_examples') else 0} negative examples."
            )
        
        return (
            grouped_results_dict, 
            cleaned_search_query_name, 
            results_with_metadata, 
            selection_results, 
            non_exact_match_results_metadata, 
            company_features, 
            negative_examples_collection
        )
    
    except Exception as e:
        logger.error(f"Error in enhanced_search_company_summary_v4: {e}")
        # Track error if tracker provided
        if tracker:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Search Error",
                status="Failed",
                details=f"Error in search process: {str(e)}"
            )
        raise

async def safe_enhanced_search_company_summary_v4(*args, **kwargs):
    """
    Wrapper around enhanced_search_company_summary_v4 with error handling and complete output.
    
    Returns:
        A 7-tuple with proper default values for any missing components
    """
    try:
        # Call the original function, not itself
        results = await enhanced_search_company_summary_v4(*args, **kwargs)
        
        # Check if results has the expected 7 items
        if isinstance(results, tuple) and len(results) == 7:
            return results
            
        # If not, pad with None values
        if isinstance(results, tuple):
            padded_results = list(results)
            while len(padded_results) < 7:
                padded_results.append(None)
            return tuple(padded_results)
            
        # If not a tuple, return a tuple of None values
        logger.error(f"Unexpected return type from enhanced_search_company_summary_v4: {type(results)}")
        return ({}, "", [], [], [], None, None)
        
    except Exception as e:
        logger.error(f"Error in enhanced_search_company_summary_v4: {str(e)}")
        logger.error(traceback.format_exc())
        return ({}, "", [], [], [], None, None)
    
async def execute_general_search(query: str, max_results: int = 5, search_depth: str = "advanced", use_cache: bool = True, force_refresh: bool = False) -> List[Dict]:
    """
    Execute a general web search with cache integration and error handling.
    """
    try:
        # Check cache first if enabled and not forcing refresh
        if use_cache and not force_refresh:
            cached_results = load_from_cache(query, "general", max_results, search_depth)
            if cached_results:
                logger.info(f"Using cached results for general query: {query}")
                
                # Add query and source type information
                for result in cached_results:
                    result['search_query'] = query
                    result['source_type'] = 'general'
                    
                return cached_results
        
        # If no cache or cache disabled, perform search
        async with sem:
            search_results_raw = await tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                include_raw_content=True
            )
            
        if not search_results_raw or not search_results_raw.get('results'):
            logger.warning(f"No results found for general query: {query}")
            return []
            
        results = search_results_raw.get('results', [])
        
        # Add query and source type information
        for result in results:
            result['search_query'] = query
            result['source_type'] = 'general'
        
        # Save to cache if enabled
        if use_cache:
            save_to_cache(query, "general", results, max_results, search_depth)
            
        return results
        
    except Exception as e:
        logger.error(f"Error in general search for '{query}': {e}")
        return []
    
async def execute_news_search(query: str, max_results: int = 5, search_depth: str = "advanced", use_cache: bool = True, force_refresh: bool = False) -> List[Dict]:
    """
    Execute a news-focused search with cache integration and error handling.
    """
    try:
        # Check cache first if enabled and not forcing refresh
        if use_cache and not force_refresh:
            cached_results = load_from_cache(query, "news", max_results, search_depth)
            if cached_results:
                logger.info(f"Using cached results for news query: {query}")
                
                # Add query and source type information
                for result in cached_results:
                    result['search_query'] = query
                    result['source_type'] = 'news'
                    
                return cached_results
        
        # If no cache or cache disabled, perform search
        async with sem:
            # Modify query to focus on news sources
            news_query = f"{query}"
            
            search_results_raw = await tavily_client.search(
                query=news_query,
                max_results=max_results,
                search_depth=search_depth,
                topic="news",
                include_answer=True,
                include_raw_content=True
            )
            
        if not search_results_raw or not search_results_raw.get('results'):
            logger.warning(f"No results found for news query: {news_query}")
            return []
            
        results = search_results_raw.get('results', [])
        
        # Add query and source type information
        for result in results:
            result['search_query'] = query
            result['source_type'] = 'news'
        
        # Save to cache if enabled
        if use_cache:
            save_to_cache(query, "news", results, max_results, search_depth)
            
        return results
        
    except Exception as e:
        logger.error(f"Error in news search for '{query}': {e}")
        return []
    
async def execute_funding_search(query: str, max_results: int = 5, search_depth: str = "advanced", use_cache: bool = True, force_refresh: bool = False) -> List[Dict]:
    """
    Execute a funding-focused search with cache integration and error handling.
    """
    try:
        # Check cache first if enabled and not forcing refresh
        if use_cache and not force_refresh:
            cached_results = load_from_cache(query, "funding", max_results, search_depth)
            if cached_results:
                logger.info(f"Using cached results for funding query: {query}")
                
                # Add query and source type information
                for result in cached_results:
                    result['search_query'] = query
                    result['source_type'] = 'funding'
                    
                return cached_results
        
        # If no cache or cache disabled, perform search
        async with sem:
            # Modify query to focus on funding information
            funding_query = f"{query} funding OR investment OR 'series' site:crunchbase.com OR site:pitchbook.com OR site:techcrunch.com"
            
            search_results_raw = await tavily_client.search(
                query=funding_query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                include_raw_content=True
            )
            
        if not search_results_raw or not search_results_raw.get('results'):
            logger.warning(f"No results found for funding query: {funding_query}")
            return []
            
        results = search_results_raw.get('results', [])
        
        # Add query and source type information
        for result in results:
            result['search_query'] = query
            result['source_type'] = 'funding'
        
        # Save to cache if enabled
        if use_cache:
            save_to_cache(query, "funding", results, max_results, search_depth)
            
        return results
        
    except Exception as e:
        logger.error(f"Error in funding search for '{query}': {e}")
        return []
    
async def execute_multi_source_search(company_name: str, use_cache: bool = True, force_refresh: bool = False) -> List[Dict]:
    """
    Execute searches across multiple sources using investment-focused queries.
    
    Args:
        company_name: Company name to search for
        use_cache: Whether to use cached results if available
        force_refresh: Whether to force a refresh of cached results
        
    Returns:
        Combined list of search results from various sources
    """
    # Check cache first
    cache_key = f"multi_source_search_{company_name}"
    if use_cache and not force_refresh:
        cached_results = load_from_cache(cache_key, "combined", None, None)
        if cached_results:
            logger.info(f"Using cached results for multi-source search: {company_name}")
            return cached_results
    
    # Generate specialized queries
    search_queries = await generate_investment_search_queries_v4(company_name)
    
    # Create search tasks
    search_tasks = []
    
    # General queries
    for query in search_queries.general_queries:
        search_tasks.append(execute_general_search(query, use_cache=use_cache, force_refresh=force_refresh))
    
    # News queries
    for query in search_queries.news_queries:
        search_tasks.append(execute_news_search(query, use_cache=use_cache, force_refresh=force_refresh))
    
    # Funding queries
    for query in search_queries.funding_queries:
        search_tasks.append(execute_funding_search(query, use_cache=use_cache, force_refresh=force_refresh))
    
    # Relationship queries
    for query in search_queries.relationship_queries:
        search_tasks.append(execute_general_search(query, use_cache=use_cache, force_refresh=force_refresh))
    
    # Execute all search tasks concurrently
    all_results_nested = await asyncio.gather(*search_tasks)
    
    # Flatten results and deduplicate
    all_results = []
    seen_urls = set()
    
    for result_list in all_results_nested:
        for result in result_list:
            url = result.get('url')
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(result)
    
    # Save results to cache if caching is enabled
    if use_cache:
        save_to_cache(cache_key, "combined", all_results, None, None)
    
    return all_results

# Web Search Agent Models and Types
class WebSearchResultItem(BaseModel):
    """Individual web search result with metadata"""
    result_title: str = Field(description="Title of the search result")
    result_content: str = Field(description="Main content or summary of the result")
    result_url: str = Field(description="URL of the source")
    result_type: str = Field(description="Type of the source (e.g., Website, News, Academic)")
    result_score: float = Field(ge=0.0, le=1.0, description="Relevance score of the result (0.0 to 1.0)")
    result_date: Optional[str] = Field(None, description="Publication or last updated date of the result")
    query_timestamp: Optional[str] = Field(default=None, description="Query Timestamp")
    search_query: Optional[str] = Field(default=None, description="Search query used to find this result")  # Added field

class WebSearchResponse(BaseModel):
    """Complete web search response including analysis"""
    search_summary: str = Field(min_length=50, description="AI-generated summary of all search results")
    search_findings: List[str] = Field(min_items=1, description="List of key findings from the search results")
    search_results: List[WebSearchResultItem] = Field(min_items=1, description="List of relevant search results")
    follow_up_queries: List[str] = Field(min_items=1, description="Suggested follow-up queries for more information")
    search_timestamp: str = Field(description="Timestamp when the search was performed")

class WebSearchParameters(BaseModel):
    """Input parameters for web search"""
    search_query: str = Field(min_length=3, description="The search query")
    max_result_count: int = Field(default=3, ge=1, le=10, description="Maximum number of results to return")
    search_date: str = Field(description="Date when search is performed")
    include_images: bool = Field(default=False, description="Whether to include image results")
    search_depth: str = Field(default="advanced", description="Search depth (basic/advanced)")

# Additional models needed for the application
class CompanyNameOutput(BaseModel):
    """Output for company name extraction."""
    company_name: Optional[str] = Field(None, description="Extracted name of the company from the text.")

class CompanyDataOutput(BaseModel):
    """Simplified output for company data extraction - Table format."""
    company_name: Optional[str] = Field(None, description="Name of the company.")
    company_url: Optional[List[str]] = Field(None, description="List of unique company website URLs.")
    product_name: Optional[List[str]] = Field(None, description="List of unique product names.")
    product_type: Optional[str] = Field(None, description="Type of product/service.")
    scientific_domain: Optional[str] = Field(None, description="Scientific domain.")
    organization_type: Optional[str] = Field(None, description="Type of organization.")
    hq_locations: Optional[List[str]] = Field(None, description="List of unique HQ locations.")
    description_abstract: Optional[str] = Field(None, description="Brief company description - summarized from all sources.")
    total_funding: Optional[str] = Field(None, description="Aggregated total funding amount (e.g., 'USD 100 Million').")
    employee_count: Optional[str] = Field(None, description="Employee count range or estimate.")
    relevant_segments: Optional[List[str]] = Field(None, description="List of unique relevant market segments.")
    investor_name: Optional[List[str]] = Field(None, description="List of unique investor names.")
    competitors: Optional[List[str]] = Field(None, description="List of unique competitor company names.")

# Define a simpler verification model for individual field verification
class SingleFieldVerificationResult(BaseModel):
    """Represents the verification result for a single field."""
    field_name: str = Field(..., description="The name of the field being verified.")
    field_value: str = Field(..., description="The stringified value of the field.")
    is_verified: bool = Field(..., description="Whether the field value is verified against the content.")
    confidence: str = Field(..., description="Confidence level: Low, Medium, High, Very High")
    notes: Optional[str] = Field(None, description="Additional verification notes or explanation.")

# Simplified agent for single field verification
single_field_verification_agent = Agent(
    model=gemini_2o_model,
    result_type=SingleFieldVerificationResult,
    system_prompt="""You are an expert at verifying if a specific piece of information is supported by a text.
    
Verify whether the given field and its value are explicitly mentioned or strongly implied in the provided content.
Focus ONLY on the specific field being verified, not on other fields or general company information.

Return a SingleFieldVerificationResult with:
- The field name (unchanged)
- The field value (as string)
- A boolean indicating if the value is verified (true) or not (false)
- Your confidence level (Low, Medium, High, Very High)
- Optional notes explaining your verification decision
"""
)
class SingleResultSelectionOutput(BaseModel):
    """Represents the output of the search result selection agent for ONE result."""
    include_in_table: bool = Field(..., description="True if this result should be included in the aggregated table.")
    reason: Optional[str] = Field(None, description="Reasoning for including or excluding this search result.")

class CompanyFeaturesOutput(BaseModel):
    """Output for company feature extraction - focusing on general overview."""
    company_overview_summary: Optional[str] = Field(None, description="Brief descriptive summary of the company.")
    industry_overview: Optional[List[str]] = Field(None, description="General industry or sector overview.")
    product_service_overview: Optional[List[str]] = Field(None, description="Overview of main products or services.")
    mission_vision_statement: Optional[str] = Field(None, description="Company's stated mission or vision.")
    target_audience_customers: Optional[List[str]] = Field(None, description="Target customers or audience.")
    technology_platform_overview: Optional[List[str]] = Field(None, description="Overview of core technology platform.")
    geographic_focus: Optional[List[str]] = Field(None, description="Geographic areas of operation or focus.")
    organization_type: Optional[List[str]] = Field(None, description="Type of organization.")

def safe_model_attr(model, attr_name, default=''):
    """Safely access an attribute from a model that might be None."""
    if model is None:
        return default
    return getattr(model, attr_name, default) if hasattr(model, attr_name) else default


#########################################
# Agent Definitions
#########################################

# Company Name Agent
company_name_agent = Agent(
    model=gemini_2o_model,
    result_type=CompanyNameOutput,
    system_prompt="""Extract the primary company name from the given text (title, URL, content).
Focus on the title and initial sentences. Return a CompanyNameOutput with the extracted name.
If no clear company name is found, set company_name to None."""
)

# Company Data Agent
company_data_agent = Agent(
    model=gemini_2o_model,
    result_type=CompanyDataOutput,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""Extract key company info for database table.
Input: extracted_company_name, search_query_company_name, text (title, URL, content).
Task: Analyze text, extract details for PRIMARY COMPANY. Use CompanyDataOutput model.
Fields: company_name, company_url, product_name, product_type, scientific_domain,
organization_type, hq_locations, description_abstract, total_funding,
employee_count, relevant_segments, investor_name, competitors.
Focus on accuracy and relevance for primary company. Leave empty if not found."""
)

# Improved Selection Agent with Framework-Based Reasoning

search_result_selection_agent = Agent(
    model=gemini_2o_model,
    result_type=SingleResultSelectionOutput,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""You are an expert AI analyst determining if a search result should be included in a company information table.

Approach this task using a structured decision framework with transparent reasoning:

## DECISION FRAMEWORK

1. Establish context understanding:
   - What is the search query company name?
   - What information do we have from exact matches and extracted content?
   - What is the source reliability hierarchy for this analysis?
   - What is the threshold for inclusion in the final table?

2. Analyze the candidate search result systematically:
   - What entity is being described in this result?
   - What evidence confirms or challenges this being about the target company?
   - What new information does this result provide beyond existing sources?
   - What contradictions exist between this result and more reliable sources?

3. Apply tiered validation criteria in sequence:

   A. ENTITY IDENTITY VALIDATION
      - Methodically assess if this result refers to the target company
      - Use structured reasoning to analyze name variations and similarities
      - Compare business domains, founding details, and other core attributes
      - Look for explicit statements linking entities if names differ significantly

   B. INFORMATION VALUE ASSESSMENT
      - If entity validation passes, assess the information contribution
      - Does this add new, relevant facts not found in existing sources?
      - Does this confirm important information from other sources?
      - Does this provide context that enhances understanding?

   C. CONSISTENCY VERIFICATION
      - Check for consistency with extracted content (highest authority)
      - Check for consistency with exact match results (high authority)
      - Identify and evaluate any contradictions with reliable sources
      - Determine if contradictions warrant exclusion or represent updates

   D. FINAL INCLUSION DECISION
      - Weigh all evidence using the full analysis above
      - Apply conservative inclusion criteria for borderline cases
      - Document clear reasoning for your decision
      - Include specific details that justify inclusion or exclusion

For your output, provide:
- include_in_table: Boolean decision based on your analysis
- reason: Detailed explanation showing your structured reasoning process"""
)

# Company Features Agent
company_features_agent = Agent(
    model=gemini_2o_model,
    result_type=CompanyFeaturesOutput,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""Extract general company features from provided context:

Analyze company name, data, and context text

Focus on broad understanding of company's purpose and characteristics

Extract information for: company overview, industry, products/services, mission/vision, target audience, technology, geographic focus, and organization type

Provide general overviews and summaries for each category

Output a CompanyFeaturesOutput object with extracted information"""
)

# Aggregated Company Data Agent
aggregated_company_data_agent = Agent(
    model=gemini_2o_model,
    result_type=CompanyDataOutput,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""Summarize and refine aggregated company data from a CompanyDataOutput object:

Create a concise description_abstract.

Aggregate total_funding and employee_count.

Ensure unique values in list fields.

Review and refine all fields for accuracy and conciseness.
Output a refined CompanyDataOutput object."""
)

# Row Review Agent
row_review_agent = Agent(
    model=gemini_2o_model,
    result_type=RowReviewResult,
    system_prompt="""You are an expert data analyst reviewing a single data point to determine if it requires additional information gathering via web search.

You will be given a 'Field Name', its current 'Value', and the corresponding 'BaseModel' (a Pydantic model defining the expected structure). Each field in the BaseModel has a 'description' explaining its purpose. You will also be given the 'Company Name'.

Your task is to determine if the current value is sufficient according to the requirements of the Pydantic BaseModel. Consider these factors:
- If the field is mandatory (not Optional) and the value is missing (None or 'nan').
- If the value is vague, incomplete, or lacks sufficient detail based on the field's 'description'.
- If the 'Value' is a URL, understand its context but focus if more context is needed beyond just the URL itself.

Identify the fields that need additional review and search to meet the requirements of their BaseModel descriptions. Return a list of these field names in the 'fields_to_review' field. If no fields need review, return an empty list for 'fields_to_review'. Set 'needs_additional_search' to True if 'fields_to_review' is not empty, otherwise False."""
)

# Suggested Query Agent
suggested_query_agent = Agent(
    model=gemini_2o_model,
    result_type=SearchQuerySuggestionResponse,
    system_prompt="""You are an expert in generating effective search queries to find information for specific fields related to companies.

Given a 'Company Name', a 'Field Name' that needs more information, and the 'description' of that field from its Pydantic BaseModel, construct a concise and effective search query.

The goal is to create a query that will help find detailed and specific information to fill the 'Field Name' for the given 'Company Name', based on the field's description.

Examples:
- Input: Company Name: 'Acme Corp', Field Name: 'publications_url', Field Description: 'URL to relevant publications about the product/service.'
Output: {"search_query": "Company Name Acme Corp: publications about products and services"}
- Input: Company Name: 'BioTech Solutions', Field Name: 'Business Model(s)', Field Description: 'List of business models.'
Output: {"search_query": "Company Name BioTech Solutions: business models"}

Focus on creating search queries that are directly relevant to the 'Field Name' and 'Company Name', utilizing keywords from the 'Field Description' to guide the search. Ensure the query starts with "Company Name {Company Name}: "."""
)

# Web Search Agent
web_search_agent = Agent(
    model=gemini_2o_model,
    deps_type=WebSearchParameters,
    result_type=WebSearchResponse,
    system_prompt=(
        "You are a web search specialist focused on accurate information retrieval and analysis.\n"
        "1. Process search results and generate a concise summary.\n"
        "2. Extract specific, actionable key findings.\n"
        "3. Evaluate and rank results by relevance.\n"
        "4. Generate targeted follow-up queries for deeper research.\n"
        "Ensure all outputs strictly follow the specified schema."
    )
)

#########################################
# Web Search API Integration
#########################################

@web_search_agent.tool
async def execute_web_search(search_context: RunContext[WebSearchParameters]) -> dict:
    """Execute web search using Tavily API with error handling."""
    start_time = datetime.now()
    try:
        search_query = search_context.deps.search_query.strip()
        if not search_query:
            raise ValueError("Search query cannot be empty")

        search_results_raw = await tavily_client.search(
            query=search_query,
            max_results=search_context.deps.max_result_count,
            search_depth=search_context.deps.search_depth,
            include_answer=True,
            include_images=search_context.deps.include_images,
            include_raw_content=True
        )

        if not search_results_raw or not search_results_raw.get('results'):
            logger.warning(f"No results found for query: {search_query}")
            return {"search_results": []}

        search_results = search_results_raw.get('results', [])
        processed_results = []
        for result in search_results:
            processed_results.append(
                WebSearchResultItem(
                    result_title=result.get("title", "Untitled"),
                    result_content=result.get("content", "No content available"),
                    result_url=result.get("url", ""),
                    result_type=result.get("type", "Website"),
                    result_score=float(result.get("score", 0.0)),
                    result_date=result.get("published_date", None),
                    query_timestamp=start_time.isoformat(),
                    search_query=search_query  # Add the search query to each result
                )
            )

        # Store raw results separately but don't return them in the WebSearchResponse
        # This avoids the Gemini API error while still preserving the data if needed
        st.session_state['last_raw_search_results'] = search_results_raw

        return {
            "search_summary": search_results_raw.get("answer", "No summary available."),
            "search_findings": [],
            "search_results": processed_results,
            "follow_up_queries": [],
            "search_timestamp": start_time.isoformat()
            # raw_search_results field removed
        }

    except Exception as e:
        error_message = f"Web search error: {str(e)}"
        logger.error(error_message)
        return {"search_results": [], "error": error_message, "search_timestamp": start_time.isoformat()}

async def process_web_search_results(search_query: str) -> Optional[WebSearchResponse]:
    """Process web search results using the agent."""
    try:
        search_params = WebSearchParameters(
            search_query=search_query,
            max_result_count=7,
            search_date=datetime.now().strftime("%Y-%m-%d"),
            include_images=False,
            search_depth="advanced"
        )
        async with sem:
            response = await web_search_agent.run(search_query, deps=search_params)
            return response.data
    except Exception as e:
        logger.error(f"Error processing web search for '{search_query}': {e}")
        return None

async def generate_multiple_search_queries(company_name: str, num_queries: int = 4) -> List[str]:
    """
    Generate multiple search queries for a company aimed at finding comprehensive information
    from various authoritative sources.
    
    Args:
        company_name: Name of the company to search for
        num_queries: Number of queries to generate (default: 4)
        
    Returns:
        List of search queries
    """
    queries = []
    
    # Always include a basic company information query
    initial_query_suggestion_output = await suggested_query_agent.run(
        user_prompt=f"Company Name: '{company_name}', Field Name: 'company information', Field Description: 'General company information search query.'"
    )
    initial_search_query = initial_query_suggestion_output.data.search_query if initial_query_suggestion_output.data else f"{company_name} company information"
    queries.append(initial_search_query)
    
    # Define targeted query templates that are likely to find high-quality sources
    query_templates = [
        f"{company_name} site:crunchbase.com company profile funding",
        f"{company_name} site:ycombinator.com",
        f"{company_name} site:linkedin.com company",
        f"{company_name} technology platform description",
        f"{company_name} founder CEO leadership team",
        f"{company_name} business model how it works",
        f"{company_name} competitors market position",
        f"{company_name} funding series investment",
        f"{company_name} products services overview",
        f"{company_name} news recent developments"
    ]
    
    # Ensure we don't exceed requested number of queries
    num_template_queries = min(num_queries - 1, len(query_templates))
    selected_templates = query_templates[:num_template_queries]
    
    # Add template-based queries
    queries.extend(selected_templates)
    
    # If we still need more queries, generate them using the agent
    if len(queries) < num_queries:
        # Topic-based queries for remaining slots
        topic_descriptions = [
            {"field": "technology", "description": "Detailed information about the company's technology, platform, or scientific approach"},
            {"field": "funding", "description": "Funding rounds, investors, and financial information"},
            {"field": "team", "description": "Founders, leadership team, and key personnel"},
            {"field": "use cases", "description": "How the company's products or services are being used in real-world applications"}
        ]
        
        # Select topics for remaining query slots
        remaining_slots = num_queries - len(queries)
        for i in range(min(remaining_slots, len(topic_descriptions))):
            topic = topic_descriptions[i]
            query_suggestion_output = await suggested_query_agent.run(
                user_prompt=f"Company Name: '{company_name}', Field Name: '{topic['field']}', Field Description: '{topic['description']}'"
            )
            search_query = query_suggestion_output.data.search_query if query_suggestion_output.data and query_suggestion_output.data.search_query else f"{company_name} {topic['field']}"
            queries.append(search_query)
    
    return queries

###############################################
# Semantic matching model and agent
###############################################

# 1. Create a data structure to store negative examples

class EnhancedNegativeExampleFeatures(BaseModel):
    """Enhanced model for negative examples with more detailed differentiation attributes."""
    entity_name: str = Field(..., description="Name of the entity that is not the target company")
    similarity_to_target: str = Field(..., description="How this entity name is similar to target")
    industry: Optional[List[str]] = Field(None, description="Industry of this non-target entity")
    founding_year: Optional[int] = Field(None, description="Year this entity was founded")
    founders: Optional[List[str]] = Field(None, description="Founders of this entity")
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    description: Optional[str] = Field(None, description="Brief description of what this entity is")
    key_differentiators: List[str] = Field(..., description="Key attributes that differentiate from target")
    confidence_score: str = Field(..., description="Confidence that this is definitely not the target: Low, Medium, High, Very High")
    source_url: Optional[str] = Field(None, description="URL where this entity was found")

class EnhancedNegativeExamplesCollection(BaseModel):
    """Enhanced collection of negative examples with metadata."""
    target_company_name: str = Field(..., description="The name of the target company")
    negative_examples: List[EnhancedNegativeExampleFeatures] = Field(default_factory=list, description="List of negative examples")
    collection_date: str = Field(..., description="Date when this collection was created")
    has_high_confidence_examples: bool = Field(False, description="Whether collection contains any high confidence examples")

# 2. Add agent to identify and extract negative examples

# Enhanced Negative Example Identification Logic

# Instead of focusing on specific examples or patterns, we'll create a framework 
# for identifying negative examples through systematic analysis

enhanced_negative_example_agent = Agent(
    model=gemini_2o_model,
    result_type=EnhancedNegativeExampleFeatures,
    system_prompt="""You are an expert at identifying when a search result is definitely NOT about the target company but about a different entity with a similar name.

Use this systematic framework to analyze potential negative examples:

## NEGATIVE EXAMPLE IDENTIFICATION FRAMEWORK

1. Baseline Analysis:
   - What is the target company we're researching?
   - What entity is being described in this search result?
   - What are the key identifying attributes of each?

2. Systematic Differentiation Analysis:
   - FOUNDING CONTEXT: Analyze founding information (who, when, where)
   - BUSINESS DOMAIN: Compare core business activities and scientific domains
   - CORPORATE IDENTITY: Evaluate legal entity details, registration, domains
   - OPERATIONAL DETAILS: Assess locations, team composition, funding history

3. For each comparison area, document:
   - What specific evidence exists in the content
   - How this evidence differentiates the entities
   - The strength and reliability of this evidence
   - Your confidence in this specific differentiation

4. Name Similarity Analysis:
   - Analyze the pattern and nature of name similarity
   - Consider industry naming conventions and disambiguation patterns
   - Evaluate if similarity is coincidental or potentially meaningful
   - Document the specific type of name relationship observed

5. Apply Objective Evidence Standards:
   - Require multiple strong differentiators for high confidence
   - Document specific textual evidence for each differentiation claim
   - Consider alternative explanations for apparent differences
   - Evaluate overall evidence strength holistically

Only produce a result if you can confidently determine this is NOT the target company.
Assign confidence levels (Low, Medium, High, Very High) based on evidence quality and quantity.
Focus on factual differentiators rather than pattern matching."""
)

async def enhanced_identify_negative_examples_v4(target_company_name, basic_company_info, search_results):
    """
    Enhanced negative example identification that extracts more detailed differentiation attributes.
    
    Args:
        target_company_name: Name of the target company
        basic_company_info: Basic information about the target company
        search_results: List of search results to analyze
        
    Returns:
        EnhancedNegativeExamplesCollection with identified negative examples
    """
    negative_examples = []
    high_confidence_found = False
    identification_tasks = []
    
    for idx, result in enumerate(search_results):
        search_result_metadata = result.get('search_result_metadata', {})
        extracted_company_name = result.get('extracted_company_name', '')
        
        # Skip obviously matching results
        if extracted_company_name and target_company_name and extracted_company_name.lower() == target_company_name.lower():
            continue
        
        # Extract name patterns that often indicate different entities
        name_similarity = ""
        if extracted_company_name and target_company_name:
            extracted_lower = extracted_company_name.lower()
            target_lower = target_company_name.lower()
            
            if extracted_lower and target_lower and extracted_lower.startswith(target_lower + " "):
                name_similarity = f"Target name '{target_lower}' with added suffix '{extracted_lower[len(target_lower)+1:]}'"
            elif target_lower.startswith(extracted_lower + " "):
                name_similarity = f"Target name '{target_lower}' is extended version of '{extracted_lower}'"
            elif extracted_lower in target_lower:
                name_similarity = f"Target name '{target_lower}' contains '{extracted_lower}'"
            elif target_lower in extracted_lower:
                name_similarity = f"Entity name '{extracted_lower}' contains target '{target_lower}'"
            else:
                similarity_score = 0
                for i in range(min(len(target_lower), len(extracted_lower))):
                    if i < len(target_lower) and i < len(extracted_lower) and target_lower[i] == extracted_lower[i]:
                        similarity_score += 1
                
                if similarity_score > 0:
                    name_similarity = f"Partial string match ({similarity_score} characters)"
                else:
                    name_similarity = "No obvious name similarity"
        
        # For each potential negative example, run the identification agent
        content = search_result_metadata.get('content', 'N/A') if search_result_metadata is not None else 'N/A'
        content_preview = content[:2500] if isinstance(content, str) else 'N/A'

        prompt = f"""
        TARGET COMPANY: {target_company_name}

        TARGET COMPANY INFO:
        {basic_company_info}

        SEARCH RESULT:
        Title: {search_result_metadata.get('title', 'N/A') if search_result_metadata is not None else 'N/A'}
        URL: {search_result_metadata.get('url', 'N/A') if search_result_metadata is not None else 'N/A'}
        Extracted Company Name: {extracted_company_name}
        Name Similarity: {name_similarity}
        Content: {content_preview}

        Determine if this search result is DEFINITELY NOT about the target company "{target_company_name}".
        If it's clearly a different entity, provide details about how it differs from the target company.
        If you're unsure or it could be the same entity, DO NOT produce a result.
        """
        
        identification_tasks.append((idx, enhanced_negative_example_agent.run(prompt)))
    
    # Run all tasks concurrently with rate limiting
    results_with_indices = []
    for idx, task in identification_tasks:
        try:
            async with sem:  # Use semaphore for rate limiting
                result = await task
                if result.data:  # Only add if the agent produced a result
                    results_with_indices.append((idx, result.data))
                    if result.data.confidence_score in ["High", "Very High"]:
                        high_confidence_found = True
        except Exception as e:
            logger.error(f"Error identifying negative example for result {idx}: {e}")
    
    # Sort results by confidence level (highest first) and limit to most confident examples
    results_with_indices.sort(key=lambda x: {"Very High": 4, "High": 3, "Medium": 2, "Low": 1}.get(x[1].confidence_score, 0), reverse=True)
    
    # Take all high/very high confidence examples and up to 5 medium/low ones
    high_confidence_examples = [r[1] for r in results_with_indices if r[1].confidence_score in ["High", "Very High"]]
    other_examples = [r[1] for r in results_with_indices if r[1].confidence_score in ["Medium", "Low"]][:5]
    
    negative_examples = high_confidence_examples + other_examples
    
    # Create and return the collection
    return EnhancedNegativeExamplesCollection(
        target_company_name=target_company_name,
        negative_examples=negative_examples,
        collection_date=datetime.now().strftime("%Y-%m-%d"),
        has_high_confidence_examples=high_confidence_found
    )


# 3. Update the semantic validation process to use negative examples

# Key Entity Attributes Models
class EntityFounderInfo(BaseModel):
    """Information about entity founders"""
    founder_names: List[str] = Field(default_factory=list, description="Names of founders")
    mentions: List[str] = Field(default_factory=list, description="Raw text mentions of founders")

class EntityTemporalInfo(BaseModel):
    """Temporal information about the entity"""
    founding_year: Optional[int] = Field(None, description="Year the entity was founded")
    acquisition_year: Optional[int] = Field(None, description="Year the entity was acquired, if applicable")
    rebranding_year: Optional[int] = Field(None, description="Year the entity was rebranded, if applicable")
    latest_known_activity: Optional[str] = Field(None, description="Latest known activity or mention with date")

class EntityLocationInfo(BaseModel):
    """Geographic information about the entity"""
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    additional_locations: List[str] = Field(default_factory=list, description="Additional office locations")

class EntityFundingInfo(BaseModel):
    """Funding information about the entity"""
    total_funding_amount: Optional[str] = Field(None, description="Total funding amount (textual representation)")
    latest_funding_round: Optional[str] = Field(None, description="Latest funding round information")
    investors: List[str] = Field(default_factory=list, description="Known investors")

class EntityProductInfo(BaseModel):
    """Product information about the entity"""
    main_products: List[str] = Field(default_factory=list, description="Main products or services")
    industry_focus: List[str] = Field(default_factory=list, description="Industry focus areas")
    technologies: List[str] = Field(default_factory=list, description="Key technologies used or developed")

class EntityRelationshipInfo(BaseModel):
    """Relationship information with other entities"""
    parent_company: Optional[str] = Field(None, description="Parent company if subsidiary")
    subsidiaries: List[str] = Field(default_factory=list, description="Known subsidiaries")
    previous_names: List[str] = Field(default_factory=list, description="Previous company names")
    related_entities: List[str] = Field(default_factory=list, description="Other related entities and their relationship")

class EntityAttributeExtraction(BaseModel):
    """Complete set of extracted attributes about an entity"""
    entity_name: str = Field(..., description="Name of the entity as mentioned in text")
    similar_names: List[str] = Field(default_factory=list, description="Similar or alternative names mentioned")
    founder_info: EntityFounderInfo = Field(default_factory=EntityFounderInfo, description="Founder information")
    temporal_info: EntityTemporalInfo = Field(default_factory=EntityTemporalInfo, description="Temporal information")
    location_info: EntityLocationInfo = Field(default_factory=EntityLocationInfo, description="Location information")
    funding_info: EntityFundingInfo = Field(default_factory=EntityFundingInfo, description="Funding information")
    product_info: EntityProductInfo = Field(default_factory=EntityProductInfo, description="Product information")
    relationship_info: EntityRelationshipInfo = Field(default_factory=EntityRelationshipInfo, description="Relationship information")

class EntityAttributeConflict(BaseModel):
    """Represents a conflict between expected and found attribute values"""
    attribute_name: str = Field(..., description="Name of the conflicting attribute")
    expected_value: str = Field(..., description="Expected value from reference entity")
    found_value: str = Field(..., description="Found value in the content being analyzed")
    conflict_severity: str = Field(..., description="Severity of the conflict: 'Low', 'Medium', 'High'")
    explanation: str = Field(..., description="Explanation of why this is a conflict")

class EnhancedEntityDifferentiation(BaseModel):
    """Enhanced model for entity differentiation analysis with weighted attributes."""
    are_same_entity: bool = Field(..., description="Whether the entities are the same or different")
    confidence_level: str = Field(..., description="Confidence level: Low, Medium, High, Very High")
    explanation: str = Field(..., description="Detailed explanation of the differentiation analysis")
    matching_attributes: List[str] = Field(default_factory=list, description="List of attributes that match between entities")
    critical_conflicts: List[str] = Field(default_factory=list, description="List of critical conflicts that strongly indicate different entities")
    high_conflicts: List[str] = Field(default_factory=list, description="List of high-importance conflicts")
    potential_relationship: Optional[str] = Field(None, description="Relationship between entities if evidence exists")
    relationship_evidence: Optional[str] = Field(None, description="Explicit evidence of the relationship in the text")
    
class StrictSemanticMatchResult(BaseModel):
    """Results from strict semantic matching with clearer entity differentiation."""
    is_valid_match: bool = Field(..., description="Whether this search result is a valid match for the target company")
    confidence_level: str = Field(..., description="Confidence level: Low, Medium, High, Very High")
    match_reasoning: str = Field(..., description="Detailed reasoning for match decision")
    is_related_entity: bool = Field(False, description="Whether this is a related but distinct entity (subsidiary, etc.)")
    relationship_type: Optional[str] = Field(None, description="Type of relationship if is_related_entity is True")
    detected_critical_conflicts: List[str] = Field(default_factory=list, description="Critical conflicts detected")
    negative_example_matches: List[str] = Field(default_factory=list, description="Negative examples matched")
    recommended_fields: List[str] = Field(default_factory=list, description="Fields that can be reliably extracted")
    warning_message: Optional[str] = Field(None, description="Warning message for potentially misleading information")

# Define the entity attribute extraction agent
# Define the entity attribute extraction agent
entity_attribute_extraction_agent = Agent(
    model=gemini_2o_model,
    result_type=FlatEntityAttributes,
    system_prompt="""You are an expert at extracting structured entity information from text. 
    
Your task is to carefully analyze content and extract key attributes about a specific entity mentioned in the text.

Focus on:
1. The entity name and any similar or alternative names 
2. Founder information (names and mentions)
3. Temporal information (founding year, acquisition year, etc.)
4. Location information (headquarters, additional locations)
5. Funding information (amounts, rounds, investors)
6. Product information (main products/services, industry focus, technologies)
7. Relationship information (parent company, subsidiaries, previous names, related entities)

Extract information into a flat structure with these fields:
- entity_name: Name of the entity as mentioned in text
- similar_names: List of similar or alternative names mentioned
- founder_names: Names of founders
- founder_mentions: Raw text mentions of founders
- founding_year: Year the entity was founded
- acquisition_year: Year the entity was acquired, if applicable
- rebranding_year: Year the entity was rebranded, if applicable
- latest_known_activity: Latest known activity or mention with date
- headquarters: Headquarters location
- additional_locations: Additional office locations
- total_funding_amount: Total funding amount (textual representation)
- latest_funding_round: Latest funding round information
- investors: Known investors
- main_products: Main products or services
- industry_focus: Industry focus areas
- technologies: Key technologies used or developed
- parent_company: Parent company if subsidiary
- subsidiaries: Known subsidiaries
- previous_names: Previous company names
- related_entities: Other related entities and their relationship

Analyze both explicit statements and implicit information. If information is missing, leave the corresponding fields empty or as default values. Do not make assumptions where information is not available.

For lists, include only clearly mentioned items, not inferences. Avoid duplicates in lists.

Return a structured FlatEntityAttributes object with all extracted information."""
)

async def extract_entity_attributes(content_text: str, entity_name: str, row_data: Optional[Dict] = None) -> FlatEntityAttributes:
    """
    Extracts structured entity attributes from content, returning a flat structure.
    Enhanced with ability to incorporate row data from Excel.
    
    Args:
        content_text: The text content to analyze
        entity_name: The primary entity name to focus on
        row_data: Optional row data from Excel to incorporate
        
    Returns:
        FlatEntityAttributes object with structured entity information
    """
    try:
        # First, prepare a prompt that includes row data if available
        row_data_str = ""
        if row_data:
            # Extract potentially useful fields from row data
            useful_fields = [
                "Year Founded", "HQ Location", "Organization Type", 
                "Scientific Domain", "Industry", "Website", "Products",
                "Founded", "Headquarters", "Funding", "Investors"
            ]
            
            extracted_fields = []
            for field in useful_fields:
                for col in row_data.keys():
                    # Check for column name matches (both exact and fuzzy)
                    if (field.lower() in col.lower() or col.lower() in field.lower()) and not pd.isna(row_data[col]):
                        extracted_fields.append(f"{col}: {row_data[col]}")
            
            if extracted_fields:
                row_data_str = "TRUSTED DATA FROM DATABASE:\n" + "\n".join(extracted_fields) + "\n\n"
        
        prompt = f"""
        {row_data_str}CONTENT TO ANALYZE:
        {content_text}
        
        Analyze the above content and extract key attributes about the entity '{entity_name}'.
        Extract all available information about '{entity_name}' following the FlatEntityAttributes schema.
        Focus only on information present in the content, do not make assumptions.
        
        If there is any conflict between the TRUSTED DATA and the content, prioritize the TRUSTED DATA as it is from a verified source.
        """
        
        result = await entity_attribute_extraction_agent.run(prompt)
        return result.data
    except Exception as e:
        # Create a minimal entity attributes object in case of error
        minimal_attributes = FlatEntityAttributes(
            entity_name=entity_name,
            similar_names=[],
            founder_names=[],
            founder_mentions=[],
            founding_year=None,
            acquisition_year=None,
            rebranding_year=None,
            latest_known_activity=None,
            headquarters=None,
            additional_locations=[],
            total_funding_amount=None,
            latest_funding_round=None,
            investors=[],
            main_products=[],
            industry_focus=[],
            technologies=[],
            parent_company=None,
            subsidiaries=[],
            previous_names=[],
            related_entities=[]
        )
        
        # If we have row data, try to add it to the minimal attributes
        if row_data:
            # Map common column names to attribute fields
            field_mappings = {
                "Year Founded": "founding_year",
                "Founded": "founding_year",
                "HQ Location": "headquarters", 
                "Headquarters": "headquarters",
                "Organization Type": "parent_company",
                "Products": "main_products"
            }
            
            for col, field in field_mappings.items():
                for row_col in row_data.keys():
                    if (col.lower() in row_col.lower() or row_col.lower() in col.lower()) and not pd.isna(row_data[row_col]):
                        # Handle different field types
                        if field == "founding_year" and isinstance(row_data[row_col], (int, float)):
                            # Convert to int for founding year
                            try:
                                minimal_attributes.founding_year = int(row_data[row_col])
                            except:
                                pass
                        elif field == "main_products" and isinstance(row_data[row_col], str):
                            # Split comma-separated product names
                            minimal_attributes.main_products = [p.strip() for p in str(row_data[row_col]).split(",")]
                        elif hasattr(minimal_attributes, field):
                            # Generic string assignment for other fields
                            setattr(minimal_attributes, field, str(row_data[row_col]))
        
        return minimal_attributes

# Enhanced Entity Differentiation Agent with Framework-Based Reasoning

enhanced_entity_differentiation_agent = Agent(
    model=gemini_2o_model,
    result_type=EnhancedEntityDifferentiation,
    system_prompt="""You are an expert at determining whether two entities with similar names are the same company or different companies.

Approach this analysis using a structured reasoning framework rather than relying on specific patterns or examples:

## ANALYSIS FRAMEWORK

1. Begin with a default position of uncertainty. Do not assume entities are the same or different based solely on name similarity.

2. Methodically evaluate evidence across these dimensions in order of importance:

   A. CORE IDENTITY ATTRIBUTES: These are primary differentiators that rarely change
      - Founders and founding team composition
      - Year of incorporation/founding 
      - Legal registration details
      - Primary business domain and scientific focus

   B. OPERATIONAL ATTRIBUTES: These can strongly indicate separate entities
      - Headquarters location and regional presence
      - Digital identity (website domains, social media handles)
      - Funding history timeline and investment rounds
      - Product/service focus and technological approach

   C. CONTEXTUAL ATTRIBUTES: Consider these with other evidence
      - Industry relationships and partnerships
      - Market positioning and competitive landscape
      - Recent organizational changes
      - Semantic context of mentions

3. When companies have similar base names with differentiating elements (suffixes, prefixes, etc.):
   - Recognize that in corporate naming, such variations are often deliberately chosen to distinguish between separate legal entities
   - Apply heightened scrutiny to all other attributes
   - Look for explicit statements about corporate relationships

4. For each attribute, apply chain-of-thought reasoning:
   - What evidence exists in the provided content?
   - What is the reliability of this evidence?
   - Does this evidence support same-entity or different-entity conclusion?
   - What weight should this evidence carry in the overall determination?

5. Evaluate potential relationships between entities:
   - Subsidiary, parent, acquisition, merger, rebranding
   - Require explicit textual evidence for any relationship claim
   - Note the date/timeline of any relationship change

6. Identify and list specific matching attributes:
   - Look for attributes that are clearly the same between both entities
   - Include matching founders, founding years, locations, business domains, etc.
   - Be specific about matches (e.g., "Same founder: John Smith" rather than just "Founders")

7. Identify and list specific differentiating attributes:
   - Critical conflicts that strongly suggest different entities
   - High importance conflicts that weigh heavily in your assessment
   - Be specific about differences (e.g., "Different founding year: 2015 vs 2018")

8. Assign confidence levels based on:
   - Quantity of evidence examined
   - Quality and reliability of evidence
   - Presence of contradictions or confirmations
   - Completeness of information

Return an EnhancedEntityDifferentiation object with:
- Your determination if entities are the same (are_same_entity)
- Your confidence level (Low, Medium, High, Very High)
- Detailed explanation showing your reasoning process
- List of matching attributes between entities
- Critical conflicts identified
- High importance conflicts identified
- Any potential relationship between entities (with evidence)"""
)


# Improved Strict Semantic Validation Agent

strict_semantic_validation_agent = Agent(
    model=gemini_2o_model,
    result_type=StrictSemanticMatchResult,
    system_prompt="""You are an expert at validating whether search results are about the target company or a different entity.

Follow this systematic framework to analyze each search result:

## VALIDATION FRAMEWORK

1. Establish a baseline understanding of the target company:
   - What is the complete target company name?
   - What key characteristics define this company? (business domain, founding, location, etc.)
   - What are the most distinctive identifying features of this company?

2. Analyze the search result in a structured manner:
   - What entity is being described in this search result?
   - What attributes are mentioned that can establish identity? (name, domain, business focus, etc.)
   - How do these attributes compare to the target company's attributes?

3. Apply critical reasoning to company name similarity:
   - When companies have similar names with variations, methodically assess:
     a) Is this likely a truncation, abbreviation, or alternate form of the same name?
     b) Or is this potentially a different entity with a similar but distinct name?
     c) What naming patterns are common in this industry sector?
     d) What evidence beyond the name supports either conclusion?

4. Conduct attribute-based verification using this hierarchy:
   - TIER 1: Core identity attributes (founders, founding date, legal entity)
   - TIER 2: Business focus and domain (products, services, industry sector, scientific domain)
   - TIER 3: Operational characteristics (location, funding, team composition)
   - TIER 4: Public presentation (website, branding, messaging)

5. For each attribute comparison, document:
   - Does this attribute match between target and result?
   - Is there a conflict that suggests different entities?
   - Is there insufficient information to judge?
   - What weight should this attribute carry in the overall assessment?

6. Consider possible entity relationships:
   - Is there evidence of acquisition, subsidiary status, rebranding, or other relationships?
   - What specific statements in the text support these relationship claims?
   - When did these relationship changes occur?

7. Conduct final validation analysis:
   - Synthesize all evidence into a holistic assessment
   - Identify the strongest evidence for and against this being a valid match
   - Apply critical thinking about what information would change your conclusion
   - Assign appropriate confidence level to your determination

Provide a StrictSemanticMatchResult with your comprehensive analysis, clearly showing how you reached your conclusion."""
)

async def strict_semantic_validation_v4(
    target_company_name: str,
    target_company_attributes: FlatEntityAttributes,
    search_result: dict,
    entity_differentiation_result: EnhancedEntityDifferentiation,
    negative_examples: List[EnhancedNegativeExampleFeatures]
) -> StrictSemanticMatchResult:
    """
    Enhanced semantic validation with improved confidence levels and relationship analysis.
    
    Args:
        target_company_name: Name of the target company
        target_company_attributes: Entity attributes of the target company
        search_result: Search result metadata
        entity_differentiation_result: Result from enhanced entity differentiation
        negative_examples: List of negative examples
        
    Returns:
        StrictSemanticMatchResult with comprehensive validation
    """
    # Basic validation checks
    if not search_result or not target_company_name:
        return StrictSemanticMatchResult(
            is_valid_match=False,
            confidence_level="Low",
            match_reasoning="Insufficient data for validation",
            is_related_entity=False,
            relationship_type=None,
            detected_critical_conflicts=[],
            negative_example_matches=[],
            recommended_fields=[],
            warning_message="Missing critical data needed for validation"
        )    
    
    try:
        # Process entity differentiation results
        differentiation_json = entity_differentiation_result.model_dump()
        
        # Extract matching attributes, critical conflicts and relationship evidence
        matching_attributes = entity_differentiation_result.matching_attributes
        critical_conflicts = entity_differentiation_result.critical_conflicts
        high_conflicts = entity_differentiation_result.high_conflicts
        relationship_evidence = entity_differentiation_result.relationship_evidence
        
        # Get search content
        content_value = search_result.get('content', '')
        search_content = content_value.lower() if isinstance(content_value, str) else ''
        content_preview = content_value[:1000] if isinstance(content_value, str) and len(content_value) > 1000 else content_value
        
        # Check for negative example matches
        negative_example_matches = []
        for example in negative_examples:
            try:
                example_name = example.entity_name if hasattr(example, 'entity_name') else ''
                if example_name and example_name.lower() in search_content:
                    # Only count as match if name appears meaningfully (more than once or in specific context)
                    occurrences = search_content.count(example_name.lower())
                    if occurrences > 1:
                        negative_example_matches.append(example_name)
                        
                        # Also check for distinguishing features mentioned
                        if hasattr(example, 'key_differentiators'):
                            for differentiator in example.key_differentiators:
                                if differentiator.lower() in search_content:
                                    # Strong evidence this is the negative example
                                    negative_example_matches.append(f"{example_name}: {differentiator}")
            except Exception as e:
                logger.error(f"Error processing negative example: {e}")
                continue
        
        # Get title and URL
        title_value = search_result.get('title', 'No title')
        url_value = search_result.get('url', 'No URL')
        
        # Enhanced validation with business domain focus
        prompt = f"""
        ENHANCED SEMANTIC VALIDATION ANALYSIS
        
        TARGET COMPANY: {target_company_name}
        
        SEARCH RESULT:
        Title: {title_value}
        URL: {url_value}
        
        ENTITY DIFFERENTIATION RESULTS:
        Matching Attributes: {matching_attributes if matching_attributes else "None identified"}
        Critical Conflicts: {critical_conflicts if critical_conflicts else "None"}
        High Importance Conflicts: {high_conflicts if high_conflicts else "None"}
        Relationship Evidence: {relationship_evidence if relationship_evidence else "None"}
        Overall Determination: {"Same entity" if entity_differentiation_result.are_same_entity else "Different entity"}
        Confidence: {entity_differentiation_result.confidence_level}
        
        NEGATIVE EXAMPLE MATCHES:
        {negative_example_matches if negative_example_matches else "None identified"}
        
        Based on this comprehensive analysis, determine:
        1. Is this search result a valid match for the target company?
        2. If not an exact match, could it be a related entity (subsidiary, acquisition, etc.)?
        3. What is the confidence level in this determination?
        4. Which fields from this result can be reliably extracted?
        
        Provide a detailed reasoning for your determination.
        """
        
        async with sem:
            result = await strict_semantic_validation_agent.run(prompt)
            return result.data
    
    except Exception as e:
        logger.error(f"Error in semantic validation v4: {str(e)}")
        return StrictSemanticMatchResult(
            is_valid_match=False,
            confidence_level="Low",
            match_reasoning=f"Error during validation: {str(e)}",
            is_related_entity=False,
            relationship_type=None,
            detected_critical_conflicts=[],
            negative_example_matches=negative_example_matches if 'negative_example_matches' in locals() else [],
            recommended_fields=[],
            warning_message="Error occurred during validation, defaulting to no match for safety."
        )

async def extract_entity_attributes(content_text: str, entity_name: str) -> EntityAttributeExtraction:
    """
    Extracts structured entity attributes from content.
    
    Args:
        content_text: The text content to analyze
        entity_name: The primary entity name to focus on
        
    Returns:
        EntityAttributeExtraction object with structured entity information
    """
    try:
        prompt = f"""
        Analyze the following content and extract key attributes about the entity '{entity_name}'.
        
        CONTENT:
        {content_text}
        
        Extract all available information about '{entity_name}' following the EntityAttributeExtraction schema.
        Focus only on information present in the content, do not make assumptions.
        """
        
        result = await entity_attribute_extraction_agent.run(prompt)
        return result.data
    except Exception as e:
        # Create a minimal entity attributes object in case of error
        minimal_attributes = EntityAttributeExtraction(
            entity_name=entity_name,
            similar_names=[],
            founder_info=EntityFounderInfo(),
            temporal_info=EntityTemporalInfo(),
            location_info=EntityLocationInfo(),
            funding_info=EntityFundingInfo(),
            product_info=EntityProductInfo(),
            relationship_info=EntityRelationshipInfo()
        )
        return minimal_attributes

def entity_attributes_compatibility_wrapper(func):
    """
    Decorator to handle compatibility between nested and flat entity attribute models.
    Ensures functions can accept either model structure.
    """
    async def wrapper(*args, **kwargs):
        # Convert any nested models to flat models in args
        new_args = []
        for arg in args:
            if hasattr(arg, 'temporal_info'):  # It's a nested model
                new_args.append(convert_to_flat_entity_attributes(arg))
            else:
                new_args.append(arg)
        
        # Convert any nested models to flat models in kwargs
        new_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, 'temporal_info'):  # It's a nested model
                new_kwargs[key] = convert_to_flat_entity_attributes(value)
            else:   
                new_kwargs[key] = value
        
        # Call the original function with converted arguments
        result = await func(*new_args, **new_kwargs)
        return result
    
    return wrapper

async def enhanced_differentiate_entities_v4(
    reference_attributes: FlatEntityAttributes, 
    comparison_attributes: FlatEntityAttributes,
    content_text: str = ""
) -> EnhancedEntityDifferentiation:
    """
    Enhanced differentiation function that uses strict rules to compare entities.
    Modified to work with flat entity attributes structure.
    """
    try:
        # Add specific context to prompt to help with differentiation
        context = ""
        
        # Extract founding years for temporal analysis
        ref_founding = reference_attributes.founding_year
        comp_founding = comparison_attributes.founding_year
        if ref_founding and comp_founding and ref_founding != comp_founding:
            context += f"\nCRITICAL TEMPORAL CONFLICT: Reference entity founded in {ref_founding}, comparison entity founded in {comp_founding}.\n"
        
        # Extract founder names for identity verification
        ref_founders = reference_attributes.founder_names
        comp_founders = comparison_attributes.founder_names
        if ref_founders and comp_founders:
            common_founders = set(f.lower() for f in ref_founders if f and isinstance(f, str)) & set(f.lower() for f in comp_founders if f and isinstance(f, str))
            if not common_founders and ref_founders and comp_founders:
                context += f"\nCRITICAL FOUNDER CONFLICT: Reference entity founders: {', '.join(ref_founders)}. Comparison entity founders: {', '.join(comp_founders)}.\n"
        
        # Look for explicit relationship evidence in the text
        relationship_phrases = [
            "acquired by", "merged with", "subsidiary of", "division of", 
            "rebranded as", "formerly known as", "parent company", 
            "changed its name to", "spun off from", "joint venture"
        ]
        
        relationship_evidence = ""
        if content_text:
            for phrase in relationship_phrases:
                if phrase in content_text.lower():
                    # Extract the sentence containing the phrase
                    content_lower = content_text.lower()
                    start_idx = max(0, content_lower.find(phrase) - 100)
                    end_idx = min(len(content_text), content_lower.find(phrase) + 100)
                    context_text = content_text[start_idx:end_idx]
                    relationship_evidence += f"RELATIONSHIP EVIDENCE FOUND: '...{context_text}...'\n"
        
        if relationship_evidence:
            context += f"\n{relationship_evidence}\n"
        
        # Format reference and comparison attributes for comparison
        ref_flat_dict = reference_attributes.model_dump()
        comp_flat_dict = comparison_attributes.model_dump()
        
        prompt = f"""
        Compare these two entities to determine if they are the same company or different companies:
        
        REFERENCE ENTITY: {reference_attributes.entity_name}
        {json.dumps(ref_flat_dict, indent=2)}
        
        COMPARISON ENTITY: {comparison_attributes.entity_name}
        {json.dumps(comp_flat_dict, indent=2)}
        
        ADDITIONAL CONTEXT:
        {context}
        
        ORIGINAL TEXT SAMPLE:
        {content_text[:1500] if content_text else "No original text provided."}
        
        Provide a detailed analysis in the EnhancedEntityDifferentiation format.
        """
        
        async with sem:
            result = await enhanced_entity_differentiation_agent.run(prompt)
            return result.data
    except Exception as e:
        logger.error(f"Error in enhanced entity differentiation: {str(e)}")
        # Return a default negative result in case of error
        return EnhancedEntityDifferentiation(
            are_same_entity=False,
            confidence_level="Low",
            explanation=f"Error during entity differentiation: {str(e)}",
            critical_conflicts=[],
            high_conflicts=[],
            potential_relationship=None,
            relationship_evidence=None
        )

@entity_attributes_compatibility_wrapper
async def strict_semantic_validation_v4(
    target_company_name: str,
    target_company_attributes: FlatEntityAttributes,
    search_result: dict,
    entity_differentiation_result: EnhancedEntityDifferentiation,
    negative_examples: List[EnhancedNegativeExampleFeatures]
) -> StrictSemanticMatchResult:
    """
    Enhanced semantic validation with improved confidence levels and relationship analysis.
    
    Args:
        target_company_name: Name of the target company
        target_company_attributes: Entity attributes of the target company
        search_result: Search result metadata
        entity_differentiation_result: Result from enhanced entity differentiation
        negative_examples: List of negative examples
        
    Returns:
        StrictSemanticMatchResult with comprehensive validation
    """
    # Basic validation checks
    if not search_result or not target_company_name:
        return StrictSemanticMatchResult(
            is_valid_match=False,
            confidence_level="Low",
            match_reasoning="Insufficient data for validation",
            is_related_entity=False,
            relationship_type=None,
            detected_critical_conflicts=[],
            negative_example_matches=[],
            recommended_fields=[],
            warning_message="Missing critical data needed for validation"
        )    
    
    try:
        # Process entity differentiation results
        differentiation_json = entity_differentiation_result.model_dump()
        
        # Extract critical conflicts and relationship evidence
        critical_conflicts = entity_differentiation_result.critical_conflicts
        high_conflicts = entity_differentiation_result.high_conflicts
        relationship_evidence = entity_differentiation_result.relationship_evidence
        
        # Get search content
        content_value = search_result.get('content', '')
        search_content = content_value.lower() if isinstance(content_value, str) else ''
        content_preview = content_value[:1000] if isinstance(content_value, str) and len(content_value) > 1000 else content_value
        
        # Check for negative example matches
        negative_example_matches = []
        for example in negative_examples:
            try:
                example_name = example.entity_name if hasattr(example, 'entity_name') else ''
                if example_name and example_name.lower() in search_content:
                    # Only count as match if name appears meaningfully (more than once or in specific context)
                    occurrences = search_content.count(example_name.lower())
                    if occurrences > 1:
                        negative_example_matches.append(example_name)
                        
                        # Also check for distinguishing features mentioned
                        if hasattr(example, 'key_differentiators'):
                            for differentiator in example.key_differentiators:
                                if differentiator.lower() in search_content:
                                    # Strong evidence this is the negative example
                                    negative_example_matches.append(f"{example_name}: {differentiator}")
            except Exception as e:
                logger.error(f"Error processing negative example: {e}")
                continue
        
        # Get title and URL
        title_value = search_result.get('title', 'No title')
        url_value = search_result.get('url', 'No URL')
        
        # Enhanced validation with business domain focus
        prompt = f"""
        ENHANCED SEMANTIC VALIDATION ANALYSIS
        
        TARGET COMPANY: {target_company_name}
        
        SEARCH RESULT:
        Title: {title_value}
        URL: {url_value}
        
        ENTITY DIFFERENTIATION RESULTS:
        Critical Conflicts: {critical_conflicts if critical_conflicts else "None"}
        High Importance Conflicts: {high_conflicts if high_conflicts else "None"}
        Relationship Evidence: {relationship_evidence if relationship_evidence else "None"}
        Overall Determination: {"Same entity" if entity_differentiation_result.are_same_entity else "Different entity"}
        Confidence: {entity_differentiation_result.confidence_level}
        
        NEGATIVE EXAMPLE MATCHES:
        {negative_example_matches if negative_example_matches else "None identified"}
        
        Based on this comprehensive analysis, determine:
        1. Is this search result a valid match for the target company?
        2. If not an exact match, could it be a related entity (subsidiary, acquisition, etc.)?
        3. What is the confidence level in this determination?
        4. Which fields from this result can be reliably extracted?
        
        Provide a detailed reasoning for your determination.
        """
        
        async with sem:
            result = await strict_semantic_validation_agent.run(prompt)
            return result.data
    
    except Exception as e:
        logger.error(f"Error in semantic validation v4: {str(e)}")
        return StrictSemanticMatchResult(
            is_valid_match=False,
            confidence_level="Low",
            match_reasoning=f"Error during validation: {str(e)}",
            is_related_entity=False,
            relationship_type=None,
            detected_critical_conflicts=[],
            negative_example_matches=negative_example_matches if 'negative_example_matches' in locals() else [],
            recommended_fields=[],
            warning_message="Error occurred during validation, defaulting to no match for safety."
        )
        
class EntityAttributeWeights:
    """Defines weights for different entity attributes during comparison."""
    # Critical attributes (strong evidence of different entities)
    FOUNDER_MISMATCH = -5.0
    FOUNDING_YEAR_MISMATCH = -4.0
    COMPANY_NAME_SUFFIX_BIOTECH = -3.5  # For "X" vs "X Bio" patterns
    COMPANY_NAME_SUFFIX_AI = -3.5       # For "X" vs "X AI" patterns
    
    # High importance attributes
    HQ_LOCATION_MISMATCH = -3.0
    CORE_BUSINESS_MISMATCH = -2.5
    FUNDING_TIMELINE_MISMATCH = -2.5
    
    # Medium importance attributes
    PRODUCT_MISMATCH = -2.0
    TECHNOLOGY_MISMATCH = -2.0
    INVESTOR_MISMATCH = -1.5
    
    # Low importance attributes
    INDUSTRY_SIMILARITY = 1.0
    GEOGRAPHIC_PROXIMITY = 0.5
    
    # Positive evidence (might be the same entity)
    EXACT_NAME_MATCH = 3.0
    WEBSITE_MATCH = 2.5
    FOUNDER_MATCH = 2.0
    FOUNDING_YEAR_MATCH = 1.5
    EXPLICIT_RELATIONSHIP_MENTION = 4.0  # e.g., "X rebranded as X Bio"

class EntityComparisonScore(BaseModel):
    """Model for entity comparison scoring results."""
    total_score: float = Field(..., description="Overall similarity score (negative suggests different entities)")
    confidence_level: str = Field(..., description="Confidence level: Low, Medium, High, Very High")
    critical_conflicts: List[str] = Field(default_factory=list, description="Critical conflicts detected")
    other_conflicts: List[str] = Field(default_factory=list, description="Other conflicts detected")
    supporting_factors: List[str] = Field(default_factory=list, description="Factors supporting entity similarity")
    suggested_relationship: Optional[str] = Field(None, description="Suggested relationship if found in text")
    detailed_scores: Dict[str, float] = Field(default_factory=dict, description="Scores by category")

def compute_entity_similarity_score(
    reference_entity: EntityAttributeExtraction,
    comparison_entity: EntityAttributeExtraction,
    content_text: str = ""
) -> EntityComparisonScore:
    """
    Computes a similarity score between two entities using a weighted scoring system.
    Negative scores suggest different entities, positive scores suggest the same entity.
    
    Args:
        reference_entity: Reference entity attributes
        comparison_entity: Entity to compare
        content_text: Optional text content for relationship detection
        
    Returns:
        EntityComparisonScore with detailed scoring breakdown
    """
    scores = {}
    critical_conflicts = []
    other_conflicts = []
    supporting_factors = []
    
    # Initialize total score
    total_score = 0.0
    
    # 1. Check company names
    ref_name = reference_entity.entity_name.lower() if reference_entity and reference_entity.entity_name else ""
    comp_name = comparison_entity.entity_name.lower() if comparison_entity and comparison_entity.entity_name else ""
    
    if ref_name == comp_name:
        total_score += EntityAttributeWeights.EXACT_NAME_MATCH
        scores["name_match"] = EntityAttributeWeights.EXACT_NAME_MATCH
        supporting_factors.append(f"Exact name match: '{ref_name}'")
    else:
        # Check for common patterns that suggest different entities
        if ref_name and comp_name and (ref_name + " bio" == comp_name or comp_name + " bio" == ref_name):
            total_score += EntityAttributeWeights.COMPANY_NAME_SUFFIX_BIOTECH
            scores["name_pattern"] = EntityAttributeWeights.COMPANY_NAME_SUFFIX_BIOTECH
            critical_conflicts.append(f"'X vs X Bio' pattern detected: '{ref_name}' vs '{comp_name}' (common pattern for different biotech companies)")
        elif ref_name and comp_name and (ref_name + " ai" == comp_name or comp_name + " ai" == ref_name):
            total_score += EntityAttributeWeights.COMPANY_NAME_SUFFIX_AI
            scores["name_pattern"] = EntityAttributeWeights.COMPANY_NAME_SUFFIX_AI
            critical_conflicts.append(f"'X vs X AI' pattern detected: '{ref_name}' vs '{comp_name}' (common pattern for different AI companies)")
    
    # 2. Check founders (critical differentiator)
    ref_founders = set(f.lower() for f in (reference_entity.founder_info.founder_names or []) if f and isinstance(f, str))
    comp_founders = set(f.lower() for f in (comparison_entity.founder_info.founder_names or []) if f and isinstance(f, str))
    
    if ref_founders and comp_founders:  # Only compare if both have founder information
        common_founders = ref_founders.intersection(comp_founders)
        if not common_founders:
            total_score += EntityAttributeWeights.FOUNDER_MISMATCH
            scores["founder_mismatch"] = EntityAttributeWeights.FOUNDER_MISMATCH
            critical_conflicts.append(f"Different founders: {', '.join(ref_founders)} vs {', '.join(comp_founders)}")
        else:
            total_score += EntityAttributeWeights.FOUNDER_MATCH
            scores["founder_match"] = EntityAttributeWeights.FOUNDER_MATCH
            supporting_factors.append(f"Common founders: {', '.join(common_founders)}")
    
    # 3. Check founding year (critical differentiator)
    ref_year = reference_entity.temporal_info.founding_year
    comp_year = comparison_entity.temporal_info.founding_year
    
    if ref_year and comp_year and ref_year != comp_year:
        # Check if the year difference is significant (>1 year suggests different companies)
        if abs(ref_year - comp_year) > 1:
            total_score += EntityAttributeWeights.FOUNDING_YEAR_MISMATCH
            scores["founding_year_mismatch"] = EntityAttributeWeights.FOUNDING_YEAR_MISMATCH
            critical_conflicts.append(f"Different founding years: {ref_year} vs {comp_year}")
    elif ref_year and comp_year and ref_year == comp_year:
        total_score += EntityAttributeWeights.FOUNDING_YEAR_MATCH
        scores["founding_year_match"] = EntityAttributeWeights.FOUNDING_YEAR_MATCH
        supporting_factors.append(f"Same founding year: {ref_year}")
    
    # 4. Check headquarters location
    ref_hq = reference_entity.location_info.headquarters
    comp_hq = comparison_entity.location_info.headquarters
    
    if ref_hq and comp_hq and isinstance(ref_hq, str) and isinstance(comp_hq, str) and ref_hq.lower() != comp_hq.lower():
        # Check if they're in completely different regions (stronger evidence of different entities)
        if not any(loc in ref_hq.lower() for loc in comp_hq.lower().split()) and not any(loc in comp_hq.lower() for loc in ref_hq.lower().split()):
            total_score += EntityAttributeWeights.HQ_LOCATION_MISMATCH
            scores["hq_location_mismatch"] = EntityAttributeWeights.HQ_LOCATION_MISMATCH
            other_conflicts.append(f"Different headquarters locations: '{ref_hq}' vs '{comp_hq}'")
    
    # 5. Check for explicit relationship mentions in the text
    suggested_relationship = None
    relationship_patterns = [
        (r"(\w+)\s+(?:was|has been)\s+(?:acquired|purchased|bought)\s+by\s+(\w+)", "acquisition"),
        (r"(\w+)\s+(?:is|became)\s+(?:a subsidiary|a division)\s+of\s+(\w+)", "subsidiary"),
        (r"(\w+)\s+(?:rebranded|changed its name)\s+(?:to|as)\s+(\w+)", "rebranding"),
        (r"(\w+)\s+(?:spun off|launched|created)\s+(\w+)", "spinoff"),
        (r"(\w+)\s+(?:merged with|joined)\s+(\w+)", "merger")
    ]
    
    if content_text:
        for pattern, rel_type in relationship_patterns:
            matches = re.findall(pattern, content_text, re.IGNORECASE)
            if matches:
                for match in matches:
                    # Check if the match involves our entities
                    if any(name.lower() in match[0].lower() or name.lower() in match[1].lower() 
                           for name in [ref_name, comp_name]):
                        suggested_relationship = rel_type
                        total_score += EntityAttributeWeights.EXPLICIT_RELATIONSHIP_MENTION
                        scores["relationship_mention"] = EntityAttributeWeights.EXPLICIT_RELATIONSHIP_MENTION
                        supporting_factors.append(f"Explicit relationship mentioned: {rel_type}")
                        break
    
    # 6. Calculate final score and confidence level
    confidence_level = "Low"
    if total_score <= -4.0:
        confidence_level = "Very High"  # Very high confidence they are different
    elif total_score <= -2.0:
        confidence_level = "High"      # High confidence they are different
    elif total_score >= 3.0:
        confidence_level = "High"      # High confidence they are the same
    elif total_score >= 1.5:
        confidence_level = "Medium"    # Medium confidence they are the same
    
    return EntityComparisonScore(
        total_score=total_score,
        confidence_level=confidence_level,
        critical_conflicts=critical_conflicts,
        other_conflicts=other_conflicts,
        supporting_factors=supporting_factors,
        suggested_relationship=suggested_relationship,
        detailed_scores=scores
    )

#########################################
# Core Processing Functions
#########################################


def extract_domain(url: str) -> str:
    """
    Extract the domain name from a URL.
    
    Args:
        url: URL string
        
    Returns:
        Domain name
    """
    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # If domain is empty, take the first part of the path
        if not domain and parsed_url.path:
            domain = parsed_url.path.split('/')[0]
            
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain or "unknown_domain"
    except:
        return "unknown_domain"

def add_validation_summary(status_container, filtered_sources, filtered_fields):
    """
    Add concise validation summary to a status container instead of detailed logs.
    
    Args:
        status_container: Streamlit container to display the summary
        filtered_sources: List of filtered source information
        filtered_fields: List of filtered field information
    """
    if not status_container:
        return
        
    with status_container:
        # Create tabs for different validation views
        validation_tab1, validation_tab2 = st.tabs(["Content Validation", "Field Validation"])
        
        with validation_tab1:
            if filtered_sources:
                # Group by content type for more concise display
                content_types = {}
                domains = {}
                
                for source in filtered_sources:
                    # Extract content type and domain
                    parts = source.split(": ", 1)
                    if len(parts) == 2:
                        url = parts[0]
                        reason_details = parts[1]
                        domain = extract_domain(url)
                        
                        # Split reason details
                        reason_parts = reason_details.split(" - ", 1)
                        if len(reason_parts) >= 1:
                            content_type = reason_parts[0]
                            
                            # Count by content type
                            if content_type not in content_types:
                                content_types[content_type] = 0
                            content_types[content_type] += 1
                            
                            # Count by domain
                            if domain not in domains:
                                domains[domain] = 0
                            domains[domain] += 1
                
                # Display summary
                st.write(f"**Summary:** Filtered {len(filtered_sources)} sources")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**By Content Type:**")
                    for content_type, count in content_types.items():
                        st.write(f"- {content_type}: {count}")
                
                with col2:
                    st.write("**By Domain:**")
                    for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"- {domain}: {count}")
                
                # Add option to view details
                with st.expander("View all filtered sources", expanded=False):
                    for source in filtered_sources:
                        st.write(f"- {source}")
            else:
                st.info("No sources were filtered due to content validation.")
        
        with validation_tab2:
            if filtered_fields:
                # Group fields by type
                field_types = {}
                reasons = {}
                
                for field_data in filtered_fields:
                    field = field_data.get('field', 'Unknown')
                    reason = field_data.get('reason', 'Unknown')
                    
                    # Count by field type
                    if field not in field_types:
                        field_types[field] = 0
                    field_types[field] += 1
                    
                    # Count by reason
                    reason_short = reason[:50] + "..." if len(reason) > 50 else reason
                    if reason_short not in reasons:
                        reasons[reason_short] = 0
                    reasons[reason_short] += 1
                
                # Display summary
                st.write(f"**Summary:** Filtered {len(filtered_fields)} fields")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**By Field Type:**")
                    for field, count in field_types.items():
                        st.write(f"- {field}: {count}")
                
                with col2:
                    st.write("**Common Reasons:**")
                    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"- {reason}: {count}")
                
                # Add option to view details
                with st.expander("View all filtered fields", expanded=False):
                    field_df = pd.DataFrame(filtered_fields)
                    st.dataframe(field_df)
            else:
                st.info("No fields were filtered due to field validation.")
                
async def process_content(content_text: str, title: str, url: str, 
                         extracted_content_gold_standard: Optional[str] = None, 
                         field_name: Optional[str] = None, 
                         search_query: Optional[str] = None,
                         tracker: Optional[BatchProcessTracker] = None,
                         company_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced version of process_content with robust error handling and retries.
    """
    try:
        result = {
            "search_result_metadata": {
                "title": title,
                "url": url,
                "content": content_text,
                "search_query": search_query or ""
            }
        }
        
        # Step 1: Extract Company Name with retry mechanism
        extracted_company_name_raw = None
        extracted_company_name = None
        max_retries = 3
        retry_count = 0
        
        # NEW: Add retry loop for company name extraction
        while retry_count < max_retries:
            try:
                async with asyncio.timeout(30):  # 30 second timeout
                    async with sem:  # Use semaphore for rate limiting
                        company_name_extraction_output = await company_name_agent.run(
                            user_prompt=f"Text:\nTitle: {title}\nURL: {url}\nContent: {content_text[:5000] if content_text else ''}"
                        )
                    extracted_company_name_raw = company_name_extraction_output.data.company_name
                    extracted_company_name = basename(extracted_company_name_raw).lower() if extracted_company_name_raw else None
                    # Success, break the loop
                    break
            except (httpx.ConnectTimeout, asyncio.TimeoutError) as e:
                retry_count += 1
                wait_time = retry_count * 2  # Exponential backoff
                logger.warning(f"Connection timeout extracting company name (attempt {retry_count}/{max_retries}). Waiting {wait_time}s: {str(e)}")
                if retry_count < max_retries:
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to extract company name after {max_retries} attempts. Fallback to title-based extraction.")
                    # Fallback: Try to extract from title if content extraction failed
                    company_from_title = title.split(' - ')[0].split(' | ')[0] if title else None
                    extracted_company_name_raw = company_from_title
                    extracted_company_name = basename(company_from_title).lower() if company_from_title else None
            except Exception as e:
                logger.error(f"Error extracting company name: {str(e)}")
                # Fallback: Use URL or title for company name
                domain = extract_domain(url)
                company_from_domain = domain.split('.')[0] if domain else None
                extracted_company_name_raw = company_from_domain or title.split(' - ')[0] if title else None
                extracted_company_name = basename(extracted_company_name_raw).lower() if extracted_company_name_raw else None
                break
        
        # Store the extracted company name, even if it's None
        result["extracted_company_name_raw"] = extracted_company_name_raw
        result["extracted_company_name"] = extracted_company_name

        # STEP 0: Validate content relevance with robust error handling
        if extracted_company_name and content_text:
            try:
                async with asyncio.timeout(25):  # 25 second timeout
                    validation_result = await content_validation_agent.run(
                        user_prompt=f"""
                        Target Company: {extracted_company_name}
                        URL: {url}
                        Title: {title}
                        Content Preview:
                        {content_text[:1500] if content_text else "No content available"}
                        
                        Is this content valid information about {extracted_company_name} or is it an access barrier/irrelevant content?
                        """
                    )
                
                # Log validation result if tracker provided, but with reduced verbosity
                if tracker and company_name and hasattr(validation_result, 'data'):
                    # Only log invalid content or high confidence validations to reduce noise
                    if not validation_result.data.is_valid or validation_result.data.confidence in ["High", "Very High"]:
                        tracker.log_content_validation(
                            company_name=company_name,
                            url=url,
                            validation_result=validation_result.data.model_dump(),
                            content_preview=content_text[:100] if content_text else ""  # Reduced preview length
                        )
                
                # Skip further processing if content is invalid with high confidence
                if hasattr(validation_result, 'data') and not validation_result.data.is_valid and validation_result.data.confidence in ["High", "Very High"]:
                    logger.info(f"Filtered out irrelevant content from {url}: {validation_result.data.content_type} - {validation_result.data.reason}")
                    
                    # Return with validation result
                    return {
                        "search_result_metadata": {
                            "title": title,
                            "url": url,
                            "content": content_text,
                            "search_query": search_query or ""
                        },
                        "extracted_company_name_raw": extracted_company_name_raw,
                        "extracted_company_name": extracted_company_name,
                        "_validation_result": validation_result.data.model_dump()
                    }
            except (httpx.ConnectTimeout, asyncio.TimeoutError) as e:
                logger.warning(f"Timeout during content validation: {str(e)}. Proceeding with extraction.")
                # Continue processing even if validation fails due to timeout
            except Exception as e:
                logger.error(f"Error during content validation: {str(e)}. Proceeding with extraction.")
                # Continue processing even if validation fails


        # Create parallel tasks for data extraction with robust error handling
        extraction_tasks = []
        
        # Step 2: Extract general company data
        prompt_text = f"Extracted Company Name: {extracted_company_name}\nTitle: {title}\nURL: {url}\n\nContent:\n{content_text}\n\nExtract company information for the PRIMARY company, which is likely '{extracted_company_name}' (if available, otherwise use search query name), discussed in the above text, title, and URL."
        extraction_tasks.append(("company_data", extract_with_timeout(company_data_agent.run(prompt_text))))
        
        # Step 3: Extract field-specific data if requested
        if field_name and field_name in COLUMN_TO_AGENT_FUNCTION:
            field_extraction_function = COLUMN_TO_AGENT_FUNCTION[field_name]
            extraction_tasks.append(("field_data", extract_with_timeout(field_extraction_function(content_text))))
        
        # Step 4: Execute all extraction tasks in parallel
        task_names = [task[0] for task in extraction_tasks]
        task_futures = [task[1] for task in extraction_tasks]
        
        extraction_results = await asyncio.gather(*task_futures, return_exceptions=True)
        
        # Track invalid fields
        invalid_fields = []
        
        # Process results with improved error handling
        for i, task_result in enumerate(extraction_results):
            task_name = task_names[i]
            
            if isinstance(task_result, Exception):
                result[f"{task_name}_error"] = str(task_result)
                logger.error(f"Error in {task_name} extraction: {str(task_result)}")
            else:
                # Successfully processed result
                if task_name == "company_data" and hasattr(task_result, 'data') and task_result.data:
                    # Extract company data with safe handling for model_dump method
                    try:
                        if hasattr(task_result.data, 'model_dump'):
                            company_data = task_result.data.model_dump() 
                        elif hasattr(task_result.data, 'dict'):
                            company_data = task_result.data.dict()
                        else:
                            company_data = task_result.data
                    except Exception as e:
                        logger.error(f"Error converting company data to dict: {str(e)}")
                        company_data = {"error": str(e)}
                    
                    # Only process field validation if we have company data and a company name
                    if company_data and extracted_company_name:
                        # Validate fields with improved error handling (using helper function)
                        invalid_fields.extend(await validate_company_data_fields(company_data, extracted_company_name, url, title, content_text))
                        
                    # Store the filtered company data
                    result["company_data"] = company_data
                    
                    # Store information about invalid fields
                    if invalid_fields:
                        result["invalid_fields"] = invalid_fields
                
                elif task_name == "field_data":
                    # For field data, handle with proper attribute checking
                    if hasattr(task_result, 'data'):
                        result["field_data"] = task_result.data
                    else:
                        result["field_data"] = task_result

        # Ensure search_query is preserved after all processing
        if "search_result_metadata" in result and search_query:
            result["search_result_metadata"]["search_query"] = search_query
            
        return result

    except Exception as e:
        logger.error(f"Error in process_content: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return basic result even in case of catastrophic failure
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "search_result_metadata": {
                "title": title,
                "url": url,
                "search_query": search_query  # Still preserve the query even in error case
            },
            "extracted_company_name": extracted_company_name if 'extracted_company_name' in locals() else None
        }

# Helper functions for robust processing

async def extract_with_timeout(task, timeout_seconds=30):
    """Wrapper to execute a task with timeout."""
    try:
        async with asyncio.timeout(timeout_seconds):
            return await task
    except asyncio.TimeoutError:
        raise TimeoutError(f"Task timed out after {timeout_seconds} seconds")

async def validate_company_data_fields(company_data, extracted_company_name, url, title, content_text):
    """Validate company data fields with robust error handling."""
    invalid_fields = []
    field_validation_tasks = []
    field_names = []
    
    # Create validation tasks for each field with data
    for field_name, field_value in company_data.items():
        if field_value is not None:  # Only validate non-None fields
            field_names.append(field_name)
            
            # Safely format field value for validation
            try:
                if isinstance(field_value, (list, dict)):
                    field_value_str = json.dumps(field_value)
                else:
                    field_value_str = str(field_value)
            except Exception as e:
                logger.error(f"Error converting field {field_name} to string: {e}")
                field_value_str = f"[Error converting to string: {type(field_value)}]"
            
            domain = extract_domain(url)
            validation_prompt = f"""
            Target Company: {extracted_company_name}
            Source Domain: {domain}
            Field Name: {field_name}
            Field Value: {field_value_str}
            Source Title: {title}
            Content Context:
            {content_text[:1500] if content_text else "No content provided"}
            
            Determine if this field value contains valid information about {extracted_company_name}.
            """
            
            # Use field_validation_with_fallback instead of direct extraction_with_timeout
            field_validation_tasks.append(field_validation_with_fallback(validation_prompt, field_name))
    
    # Execute all validation tasks in parallel
    if field_validation_tasks:
        validation_results = await asyncio.gather(*field_validation_tasks, return_exceptions=True)
        
        # Process validation results
        for j, validation_result in enumerate(validation_results):
            if j >= len(field_names):
                continue
                
            field_name = field_names[j]
            
            if isinstance(validation_result, Exception):
                logger.error(f"Error validating field {field_name}: {str(validation_result)}")
                continue
                
            # If field is invalid with high confidence, mark it for removal
            if (hasattr(validation_result, 'data') and 
                    not validation_result.data.is_valid and 
                    validation_result.data.confidence in ["High", "Very High"]):
                invalid_fields.append({
                    "field": field_name,
                    "reason": validation_result.data.reason,
                    "confidence": validation_result.data.confidence
                })
                # Remove the invalid field from company data
                company_data.pop(field_name, None)
    
    return invalid_fields

async def get_aggregated_company_data(company_data_for_table: List[Dict[str, Any]], from_exact_match: bool = True) -> Tuple[CompanyDataOutput, bool]:
    """
    Aggregates a list of CompanyDataOutput dictionaries into a single CompanyDataOutput object.
    Returns a tuple with the aggregated data and a flag indicating if it's from exact matches.
    """
    aggregated_data = {}
    fields_to_aggregate = [
        "company_name", "company_url", "product_name", "product_type",
        "scientific_domain", "organization_type", "hq_locations",
        "description_abstract", "total_funding", "employee_count",
        "relevant_segments", "investor_name", "competitors"
    ]
    
    for field in fields_to_aggregate:
        aggregated_data[field] = None
    
    for company_data in company_data_for_table:
        for field in fields_to_aggregate:
            value = company_data.get(field)
            if value:
                if field in ["product_name", "hq_locations", "relevant_segments", "investor_name", "competitors", "company_url"]:
                    if aggregated_data[field] is None:
                        aggregated_data[field] = []
                    if isinstance(value, list):
                        aggregated_data[field].extend([item for item in value if item is not None and item not in (aggregated_data[field] or [])])
                    else:
                        if value not in (aggregated_data[field] or []):
                            aggregated_data[field].append(value)
                else:
                    if aggregated_data[field] is None:
                        aggregated_data[field] = value
    
    # Convert lists to sets to ensure uniqueness and then back to lists
    for field in ["company_url", "product_name", "hq_locations", "relevant_segments", "investor_name", "competitors"]:
        if aggregated_data[field] is not None:
            aggregated_data[field] = list(set(aggregated_data[field]))
    
    return CompanyDataOutput(**aggregated_data), from_exact_match

async def review_data_row(company_name: str, row_data: dict, sem: asyncio.Semaphore) -> RowReviewResult:
    """
    Reviews each field of a row against its Pydantic model concurrently.
    Identifies fields that need additional search.
    """
    fields_to_review = []
    review_tasks = []
    field_names_to_process = []
    
    for row_data_field_name, value in row_data.items():
        if row_data_field_name in COLUMN_TO_MODEL and COLUMN_TO_MODEL[row_data_field_name] != str:
            model = COLUMN_TO_MODEL[row_data_field_name]
            model_name = model.__name__
            
            list_of_model_field_details = []
            if hasattr(model, 'model_fields'):
                for model_field_name, model_field_info in model.model_fields.items():
                    description = model_field_info.description
                    list_of_model_field_details.append({
                        "model_name": model_name,
                        "field_name": model_field_name,
                        "description": description
                    })
            
            user_prompt_parts = [
                f"Field Name: '{row_data_field_name}'",
                f"Company Name: '{company_name}'",
                f"Value: '{value}'",
                f"BaseModel: {model_name}",
                f"Field Descriptions: {list_of_model_field_details}"
            ]
            user_prompt = ", ".join(user_prompt_parts)
            
            async def run_agent_with_semaphore(prompt):
                async with sem:
                    return await row_review_agent.run(prompt)
            
            review_tasks.append(run_agent_with_semaphore(user_prompt))
            field_names_to_process.append(row_data_field_name)
    
    field_review_outputs = await asyncio.gather(*review_tasks)
    
    for i, row_data_field_name in enumerate(field_names_to_process):
        agent_output = field_review_outputs[i]
        if agent_output.data.needs_additional_search:
            fields_to_review.append(row_data_field_name)
    
    needs_additional_search = len(fields_to_review) > 0
    return RowReviewResult(fields_to_review=fields_to_review, needs_additional_search=needs_additional_search)

async def handle_field_extraction(extraction_function, content):
    """
    Improved wrapper for field extraction functions with better error handling.
    
    Args:
        extraction_function: The extraction function to call
        content: The content to extract from
        
    Returns:
        The extracted data model directly
    """
    if not content or not isinstance(content, str) or len(content) < 10:
        logger.warning("Content too short or invalid type for field extraction")
        return None
        
    try:
        logger.debug(f"Extracting field using function: {extraction_function.__name__}")
        
        # Apply a timeout to prevent hanging extractions
        async def extraction_with_timeout():
            return await extraction_function(content)
        
        # Set a reasonable timeout (30 seconds) for extractions
        result = await asyncio.wait_for(extraction_with_timeout(), timeout=30)
        
        # Handle both return types:
        # 1. If it's a RunResult object with a .data attribute
        if hasattr(result, 'data'):
            data = result.data
            logger.debug(f"Extraction returned RunResult with data: {type(data)}")
            return data
        # 2. If it's already the model object itself
        elif result is not None:
            logger.debug(f"Extraction returned direct model: {type(result)}")
            return result
        else:
            logger.warning("Extraction function returned None")
            return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout in field extraction for {extraction_function.__name__}")
        return None
    except Exception as e:
        logger.error(f"Error in field extraction: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Add these improved field validation functions:

async def field_validation_with_fallback(validation_prompt, field_name):
    """
    Perform field validation with fallback mechanisms for error handling.
    
    Args:
        validation_prompt: The prompt for field validation
        field_name: The name of the field being validated
        
    Returns:
        Validation result or default result if validation fails
    """
    try:
        # Apply a timeout to prevent hanging validation
        async def validation_with_timeout():
            return await field_validation_agent.run(validation_prompt)
        
        # Set a reasonable timeout (20 seconds) for validation
        result = await asyncio.wait_for(validation_with_timeout(), timeout=20)
        return result
    except asyncio.TimeoutError:
        logger.error(f"Timeout in field validation for {field_name}")
        # Return a default "valid" result to avoid filtering out due to timeout
        return type('ValidationResult', (), {
            'data': type('ValidationData', (), {
                'is_valid': True,
                'confidence': "Low",
                'reason': "Validation timed out, default to valid"
            })
        })
    except Exception as e:
        logger.error(f"Error validating field {field_name}: {str(e)}")
        # Return a default "valid" result to avoid filtering out due to error
        return type('ValidationResult', (), {
            'data': type('ValidationData', (), {
                'is_valid': True,
                'confidence': "Low",
                'reason': f"Validation error: {str(e)}"
            })
        })

# Sanitize function for cleaning DataFrame column values
def sanitize_dataframe_values(df):
    """
    Sanitize values in a DataFrame to ensure PyArrow compatibility.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        DataFrame with sanitized values
    """
    clean_df = df.copy()
    
    # Identify columns with potential mixed types
    problem_columns = []
    for col in clean_df.columns:
        # Check for columns with complex or mixed data types
        if clean_df[col].dtype == 'object':
            has_list = clean_df[col].apply(lambda x: isinstance(x, (list, tuple))).any()
            has_dict = clean_df[col].apply(lambda x: isinstance(x, dict)).any()
            
            if has_list or has_dict:
                problem_columns.append(col)
    
    # Process problematic columns
    for col in problem_columns:
        # Convert to string representation
        clean_df[col] = clean_df[col].apply(
            lambda x: json.dumps(x) if isinstance(x, (list, dict, tuple)) else 
                     (str(x) if x is not None else None)
        )
    
    return clean_df


def add_download_button(final_df: pd.DataFrame):
    """
    Adds Excel and CSV download buttons with better error handling.
    """
    if final_df is not None:
        try:
            # First try Excel
            output = io.BytesIO()
            
            # Sanitize DataFrame for Excel export
            safe_df = sanitize_dataframe_values(final_df)
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                safe_df.to_excel(writer, index=False)
                
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                for col_num, value in enumerate(safe_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                for i, col in enumerate(safe_df.columns):
                    max_len = max(safe_df[col].astype(str).apply(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, max_len)
            
            output.seek(0)
            st.download_button(
                label="Download Full Report (Excel)",
                data=output,
                file_name="company_analysis_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error preparing Excel file: {str(e)}")
        
        # Always provide CSV as a reliable option
        try:
            csv_data = final_df.to_csv(index=False)
            st.download_button(
                label="Download Full Report (CSV)",
                data=csv_data,
                file_name="company_analysis_report.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error preparing CSV file: {str(e)}")


def highlight_non_exact_match_row(row):
    """Highlights the row in yellow if 'Non-Exact Match' is True, otherwise no highlight."""
    is_non_exact = row['Data Source'] == 'Potentially Relevant (Non-Exact Match)'
    return ['background-color: yellow' if is_non_exact else '' for _ in row.index]

async def display_table_results(company_data_for_table: List[Dict[str, Any]], cleaned_search_query_name: str, search_result_metadata_list):
    """
    Displays the aggregated company data in a table format (single row) and source list.
    Uses the aggregated agent to refine and summarize the data.
    """
    table_container = st.container()
    with table_container:
        if cleaned_search_query_name and cleaned_search_query_name != "unknown entity":
            entity_header_name = cleaned_search_query_name.title()
        else:
            entity_header_name = "Company Information"
        
        st.header(f"Entity: {entity_header_name} (Aggregated & Summarized from Selected Sources)")
        st.warning("AI-Generated Company Data: Please verify critical information with official sources.")
        
        if company_data_for_table:
            aggregated_company_data_object, from_exact_match = await get_aggregated_company_data(company_data_for_table)
            
            with st.spinner("Summarizing and refining aggregated data..."):
                try:
                    aggregated_agent_output = await aggregated_company_data_agent.run(
                        user_prompt=f"Aggregated Company Data:\n{aggregated_company_data_object.model_dump_json()}",
                    )
                    final_company_data = aggregated_agent_output.data
                except Exception as e:
                    st.error(f"Error during aggregated data summarization: {e}")
                    final_company_data = aggregated_company_data_object
            
            df_aggregated_final = pd.DataFrame([final_company_data.model_dump()])
            df_aggregated_final.columns = [col.replace('_', ' ').title() for col in df_aggregated_final.columns]
            
            data_source_label = 'Exact Match' if from_exact_match else 'Potentially Relevant (Non-Exact Match)'
            df_aggregated_final['Data Source'] = data_source_label
            
            if not from_exact_match:
                styled_df = df_aggregated_final.style.apply(highlight_non_exact_match_row, axis=1)

                st.dataframe(styled_df)
            else:
                st.dataframe(df_aggregated_final)
            
            st.info("Rows highlighted in yellow indicate data aggregated from potentially relevant (non-exact match) sources. Manual review on the 'Results' tab is recommended.")
            
            st.markdown("For detailed results and manual review, please check the **Results** tab.")
            
            st.markdown("**Sources Used for Aggregation (Selected Results):**")
            for metadata in search_result_metadata_list:
                st.write(f"- [{metadata['title']}]({metadata['url']})")
        else:
            st.info("No company data to display in table format (from selected results).")

def display_exact_match_aggregated_table(company_names, aggregated_data_list, from_exact_match_list, original_inputs_list):
    """
    Displays a single aggregated table for exact match results of all companies on the main tab.
    Highlights rows based on whether they come from exact or non-exact matches.
    """
    exact_match_table_container = st.container()
    with exact_match_table_container:
        st.subheader(f"Aggregated Company Information Table for: {', '.join([name.title() for name in company_names])}")
        if aggregated_data_list:
            df_aggregated_all_companies_raw = pd.DataFrame([data.model_dump() for data in aggregated_data_list])
            df_aggregated_all_companies_raw.columns = [col.replace('_', ' ').title() for col in df_aggregated_all_companies_raw.columns]
            
            df_aggregated_all_companies_raw.insert(0, "Original Input", original_inputs_list)
            
            data_source_labels = ['Exact Match' if is_exact else 'Potentially Relevant (Non-Exact Match)' for is_exact in from_exact_match_list]
            df_aggregated_all_companies_raw['Data Source'] = data_source_labels
            
            styled_all_companies_df = df_aggregated_all_companies_raw.style.apply(highlight_non_exact_match_row, axis=1)
            st.dataframe(styled_all_companies_df)
            
            st.info("Rows highlighted in yellow indicate data aggregated from potentially relevant (non-exact match) sources. Manual review on the 'Results' tab is recommended.")
            
            st.markdown("For detailed results and manual review, please check the **Results** tab.")
        else:
            st.warning(f"No exact match results found for {', '.join(company_names)}. Please review results on the 'Results' tab for manual selection and aggregation.")

class ContentValidationResult(BaseModel):
    """Result of validating if content is relevant to the target company."""
    is_valid: bool = Field(..., description="Whether the content contains valid information about the target company")
    content_type: str = Field(..., description="Type of content: VALID_COMPANY_INFO, SECURITY_BLOCK, PAYWALL, ERROR_PAGE, IRRELEVANT")
    confidence: str = Field(..., description="Confidence level: Low, Medium, High, Very High")
    reason: Optional[str] = Field(None, description="Reason if content is invalid")

content_validation_agent = Agent(
    model=gemini_2o_model,
    result_type=ContentValidationResult,
    system_prompt="""You are an expert at identifying irrelevant or blocked content when researching companies.

Your task is to determine if the provided content actually contains relevant information about the target company, or if it's an access barrier, error message, or unrelated content.

Analyze the content and classify it into one of these categories:
- VALID_COMPANY_INFO: Contains actual information about the target company
- SECURITY_BLOCK: Contains security messages (Cloudflare, CAPTCHA challenges)
- PAYWALL: Contains subscription requirements or paywall notices
- ERROR_PAGE: Contains error messages or access denied information
- IRRELEVANT: Content not related to the target company

Common patterns to watch for:
1. Security blocks often mention "protecting websites", "security challenge", or "check your browser"
2. Paywalls typically include phrases like "subscribe", "to continue reading", or "premium content"
3. Error pages mention "access denied", "404", "not found" or "unavailable"
4. Generic information pages describe general concepts without mentioning the target company

Domain-specific patterns to detect:
- Security services descriptions usually mention "protecting websites from attacks"
- Generic business structure information usually includes standard definitions like "sole proprietorship"
- Financial service descriptions like "M&A Advisory Services" from non-company domains
- General news subscription information like "Global markets news"

These patterns strongly indicate the content is not about the specific target company.

Return a ContentValidationResult with:
- is_valid: true ONLY if the content contains actual information about the target company
- content_type: The classification from the categories above
- confidence: Your confidence level in this classification
- reason: Brief explanation if the content is invalid
"""
)

class FieldValidationResult(BaseModel):
    """Result of validating if specific fields contain relevant information about the target company."""
    field_name: str = Field(..., description="Name of the field being validated")
    is_valid: bool = Field(..., description="Whether this field contains valid information about the target company")
    confidence: str = Field(..., description="Confidence level: Low, Medium, High, Very High")
    reason: Optional[str] = Field(None, description="Reason if field content is invalid")

field_validation_agent = Agent(
    model=gemini_2o_model,
    result_type=FieldValidationResult,
    system_prompt="""You are an expert at identifying irrelevant or incorrect information in specific data fields during company research.

Your task is to analyze whether a specific field contains relevant information about the target company, or if it contains irrelevant, generic, or self-referential information.

Analyze the field data for these semantic patterns:

1. SELF-DESCRIPTION: Content where the source describes itself rather than the target company
   - Sources describing their own products/services as if they were the company's
   - Professional service firms listing their own services
   - Academic institutions describing their courses

2. GENERIC INFORMATION: Content providing general business knowledge not specific to the target
   - Standard business structures or processes unrelated to the specific company
   - General industry descriptions without company-specific details
   - Generic regulatory or legal information

3. TEMPORAL INCONSISTENCY: Content with implausible or contradictory dates
   - Future founding dates
   - Dates that are mathematically impossible or implausible
   - Dates that contradict established company timelines

4. FIELD MISMATCHES: Content where field data is categorically inappropriate
   - Listing generic categories as specific products
   - Using entity names as relationship descriptions
   - Confusing organizational entities with individuals

Analyze the given field value in context of the target company and determine if it contains valid information specifically about that company.

Return a FieldValidationResult with:
- field_name: The name of the field being validated
- is_valid: true only if the field contains valid information about the target company
- confidence: Your confidence level in this determination
- reason: Brief explanation if the field is invalid
"""
)

async def process_single_truth_field(field_name: str, field_data_list: List[Dict]) -> Optional[Dict]:
    """
    Process a single-truth field by resolving multiple sources to a single reliable value.
    
    Args:
        field_name: Name of the field (e.g., "Year Founded")
        field_data_list: List of extracted field data with sources
        
    Returns:
        Dictionary with the resolved field value and metadata
    """
    if not field_data_list:
        return None
    
    try:
        # Format results for the agent
        formatted_results = []
        for item in field_data_list:
            data = item.get('data')
            source = item.get('source', 'Unknown')
            
            # Handle different data types appropriately
            value = None
            if hasattr(data, 'model_dump'):
                value = data.model_dump()  
            elif isinstance(data, dict):
                value = data
            elif data is not None:
                # Extract the main value based on field type
                if field_name == "Year Founded" and hasattr(data, 'year_founded'):
                    value = data.year_founded
                elif field_name == "HQ Location" and hasattr(data, 'locations') and data.locations:
                    value = data.locations[0] if isinstance(data.locations, list) else data.locations
                elif field_name == "Organization Type" and hasattr(data, 'organization_type'):
                    value = data.organization_type
                else:
                    value = str(data)
            
            if value is not None:
                formatted_results.append({
                    "value": value,
                    "source": source
                })
        
        if not formatted_results:
            return None
            
        # Use the single result agent to determine the most reliable result
        aggregation_prompt = f"""
        Field name: {field_name}
        Results from multiple sources: {json.dumps(formatted_results, indent=2)}
        
        Analyze these results and determine the most reliable value for this field.
        Provide your chain-of-thought reasoning, conclusion, confidence level, and supporting sources.
        """
        
        aggregation_result = await single_result_agent.run(aggregation_prompt)
        
        if aggregation_result and aggregation_result.data:

            # In process_single_truth_field, convert conclusion to the appropriate type when necessary
            value = aggregation_result.data.conclusion if aggregation_result and aggregation_result.data else "Unknown"

            # If the field expects a specific type (like int for Year Founded), convert it
            if field_name == "Year Founded" and value not in ["Unknown", None]:
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass  # Keep as string if conversion fails

            # Return the resolved result
            return {
                'field_name': field_name,
                'value': value,
                'confidence': aggregation_result.data.confidence if aggregation_result and aggregation_result.data else "Low",
                'supporting_sources': aggregation_result.data.supporting_sources if aggregation_result and aggregation_result.data else [],
                'reasoning': aggregation_result.data.reasoning if aggregation_result and aggregation_result.data else "No reasoning available"
            }
        
        return None
    
    except Exception as e:
        logger.error(f"Error processing single-truth field {field_name}: {e}")
        return None

async def execute_specific_field_deep_search(query: str, max_results: int = 5, tracker=None, company_name=None) -> List[Dict]:
    """
    Execute a web search for additional field research with tracker integration.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        tracker: Optional BatchProcessTracker for tracking
        company_name: Optional company name for tracking
        
    Returns:
        List of search results
    """
    try:
        # Log search start if tracker provided
        start_time = time.time()
        if tracker and company_name:
            tracker.log_search_query(company_name, query, start_time)
        
        async with sem:  # Use existing semaphore for rate limiting
            search_results_raw = await tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True
            )
            
        if not search_results_raw or not search_results_raw.get('results'):
            logger.warning(f"No results found for query: {query}")
            
            # Log empty results if tracker provided
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Field Search",
                    status="No Results",
                    details=f"No results found for query: {query}",
                    search_query=query,
                    duration=time.time() - start_time
                )
            return []
            
        results = search_results_raw.get('results', [])
        
        # Log each result if tracker provided
        if tracker and company_name:
            for result in results:
                tracker.log_search_result(
                    company_name=company_name,
                    result=result,
                    query=query
                )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in search for '{query}': {e}")
        
        # Log error if tracker provided
        if tracker and company_name:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Field Search",
                status="Error",
                details=f"Error in search: {str(e)}",
                search_query=query
            )
        return []


async def process_multi_source_field(field_name: str, field_data_list: List[Dict], 
                                    company_name: str, do_additional_research: bool = False,
                                    tracker=None, tracking_company_name=None) -> Optional[Dict]:
    """
    Process a multi-source field by aggregating values from multiple sources.
    Optionally performs additional research for enrichment.
    
    Args:
        field_name: Name of the field (e.g., "Products / Services")
        field_data_list: List of extracted field data with sources
        company_name: Name of the company
        do_additional_research: Whether to perform additional research
        tracker: Optional BatchProcessTracker for tracking
        tracking_company_name: Optional company name for tracking
        
    Returns:
        Dictionary with aggregated field values and metadata
    """
    if not field_data_list:
        return None
    
    try:
        # Extract and deduplicate values
        unique_values = {}  # value -> [sources]
        
        for item in field_data_list:
            if not isinstance(item, dict) or not item.get('data'):
                continue
                
            data = item.get('data')
            source = item.get('source', 'Unknown')
            
            # Special handling for funding rounds
            if field_name == "Funding Rounds" and data:
                if hasattr(data, 'funding_rounds'):
                    for round_info in data.funding_rounds:
                        # Create formatted funding round string
                        formatted_value = f"{company_name} Funding Round: "
                        if round_info.round_date:
                            formatted_value += f"{round_info.round_date}"
                        if round_info.amount:
                            formatted_value += f" - {round_info.amount}"
                        if round_info.round_type:
                            formatted_value += f" - {round_info.round_type} Round"
                        
                        # Use round date as key for deduplication
                        round_key = f"{round_info.round_date or ''}-{round_info.round_type or ''}"
                        
                        if round_key not in unique_values:
                            unique_values[round_key] = {
                                'value': formatted_value,
                                'sources': [source],
                                'raw_data': round_info
                            }
                        else:
                            if source not in unique_values[round_key]['sources']:
                                unique_values[round_key]['sources'].append(source)
                        
                        # Also create investor entries linked to this round
                        if round_info.investors:
                            investor_value = f"{company_name} Investors: {round_info.round_date} - "
                            investor_value += ", ".join(round_info.investors)
                            investor_value += f" - {round_info.round_type} Round"
                            
                            investor_key = f"investors-{round_key}"
                            if investor_key not in unique_values:
                                unique_values[investor_key] = {
                                    'value': investor_value,
                                    'sources': [source],
                                    'raw_data': round_info.investors
                                }
                            else:
                                if source not in unique_values[investor_key]['sources']:
                                    unique_values[investor_key]['sources'].append(source)
                
                # Add total funding if available
                if hasattr(data, 'total_funding') and data.total_funding:
                    total_key = "total_funding"
                    formatted_total = f"{company_name} Total Funding: {data.total_funding}"
                    
                    if total_key not in unique_values:
                        unique_values[total_key] = {
                            'value': formatted_total,
                            'sources': [source],
                            'raw_data': {'total_funding': data.total_funding}
                        }
                    else:
                        if source not in unique_values[total_key]['sources']:
                            unique_values[total_key]['sources'].append(source)
            
            # Enhanced handling for investors
            elif field_name == "Investors" and data:
                if hasattr(data, 'investors'):
                    for idx, investor in enumerate(data.investors):
                        if not hasattr(investor, 'investor_name') or not investor.investor_name:
                            continue
                            
                        # Create formatted investor string with investment details
                        investor_key = f"investor-{investor.investor_name.lower()}"
                        
                        # Format with investment date and round if available
                        formatted_value = f"{company_name} Investor: "
                        
                        # Add investment date if available
                        investment_date = getattr(investor, 'investment_date', None)
                        if investment_date:
                            formatted_value += f"{investment_date} - "
                        
                        # Add investor name
                        formatted_value += investor.investor_name
                        
                        # Add investment stage/round if available
                        investment_stage = getattr(investor, 'investment_stage', None)
                        if investment_stage:
                            formatted_value += f" - {investment_stage}"
                        
                        if investor_key not in unique_values:
                            unique_values[investor_key] = {
                                'value': formatted_value,
                                'sources': [source],
                                'raw_data': investor
                            }
                        else:
                            if source not in unique_values[investor_key]['sources']:
                                unique_values[investor_key]['sources'].append(source)
            
            # Existing handling for other field types
            # [Original code for other field types would remain here]
            else:
                # Extract the main finding from the data
                finding = extract_main_finding(field_name, data) if 'extract_main_finding' in globals() else str(data)
                
                if finding:
                    value_str = str(finding).strip()
                    if value_str in unique_values:
                        if source not in unique_values[value_str]['sources']:
                            unique_values[value_str]['sources'].append(source)
                    else:
                        unique_values[value_str] = {
                            'value': value_str,
                            'sources': [source],
                            'raw_data': data
                        }
                        
                    # Log extraction if tracker provided
                    if tracker and tracking_company_name:
                        tracker.log_extraction(
                            company_name=tracking_company_name,
                            field_name=field_name,
                            extracted_value=value_str,
                            success=True,
                            source_url=source
                        )
        
        # Prepare the final result
        aggregated_values = []
        aggregated_sources = []
        
        for key, item in unique_values.items():
            aggregated_values.append(item['value'])
            for source in item['sources']:
                if source not in aggregated_sources:
                    aggregated_sources.append(source)
        
        # Determine confidence based on source count and consistency
        confidence = "Low"
        if len(aggregated_sources) >= 3:
            confidence = "High"
        elif len(aggregated_sources) >= 2:
            confidence = "Medium"
            
        result = {
            'field_name': field_name,
            'values': aggregated_values,
            'sources': aggregated_sources,
            'confidence': confidence
        }
        
        # Log final result if tracker provided
        if tracker and tracking_company_name:
            tracker.add_data_entry(
                company_name=tracking_company_name,
                operation="Aggregate Field",
                status="Completed",
                details=f"Aggregated {len(aggregated_values)} values for {field_name} from {len(aggregated_sources)} sources (Confidence: {confidence})"
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing multi-source field {field_name}: {e}")
        # Log error if tracker provided
        if tracker and tracking_company_name:
            tracker.add_data_entry(
                company_name=tracking_company_name,
                operation="Aggregate Field",
                status="Error",
                details=f"Error processing field {field_name}: {str(e)}"
            )
        return None

# Pydantic Models
class SingleResultValidation(BaseModel):
    """Reasoning through results from multiple sources to determine a single reliable result."""
    field_name: str = Field(..., description="Name of the field being validated")
    reasoning: str = Field(..., description="Chain of thought reasoning about the sources and values")
    conclusion: str = Field(..., description="The final validated field value")
    confidence: str = Field(..., description="Overall confidence: Low, Medium, High, Very High")
    supporting_sources: List[str] = Field(default_factory=list, description="List of URLs supporting the conclusion")

class SpecificFieldAdditionalSearchQuerySuggestion(BaseModel):
    """Suggested search query for additional research."""
    field_name: str = Field(..., description="Field name the query is for")
    field_value: str = Field(..., description="Field value to research")
    search_query: str = Field(..., description="Suggested search query")

class SearchAnalysisResult(BaseModel):
    """Analysis of additional search results for in-depth details."""
    detailed_description: str = Field(..., description="Detailed description of the finding")
    key_points: List[str] = Field(default_factory=list, description="List of key points extracted from additional research")
    additional_search_terms: List[str] = Field(default_factory=list, description="Suggested additional search terms for more depth")
    supporting_sources: List[str] = Field(default_factory=list, description="List of sources supporting the conclusion")

# Field classification constants 
SINGLE_TRUTH_FIELDS = [
    "Year Founded", 
    "HQ Location", 
    "Organization Type", 
    "Scientific Domain", 
    "Funding Stage", 
    "Female Co-Founder"
]

MULTI_SOURCE_RESEARCH_FIELDS = [
    "Products / Services", 
    "Publicly Announced Partnerships", 
    "Investors", 
    "Advisor / Board Members / Key People", 
    "Business Model(s)"
]

# Single result validation agent for aggregating single truth fields
single_result_agent = Agent(
    model=gemini_2o_model,
    result_type=SingleResultValidation,
    system_prompt="""You are an expert at resolving conflicts and determining the most reliable value from multiple sources.
For fields with a single source of truth (like founding year, HQ location), your task is to:
1. Carefully analyze all values from different sources
2. Evaluate the reliability of each source
3. Consider the evidence and context for each value
4. Provide step-by-step reasoning about which value is most reliable
5. Determine the single most accurate conclusion

Focus on:
- Consistency across sources (do multiple reliable sources agree?)
- Source credibility (is one source particularly authoritative?)
- Evidence quality (does one value have stronger supporting evidence?)
- Recency and specificity (is one source more current or detailed?)

Provide clear chain-of-thought reasoning showing how you weighed different factors.
Your reasoning should explicitly show how you arrived at your final conclusion.

Return a SingleResultValidation with:
- field_name: The name of the field
- reasoning: Detailed step-by-step reasoning process
- conclusion: The final determined value
- confidence: Overall confidence level (Low, Medium, High, Very High)
- supporting_sources: List of URLs that support your conclusion
"""
)

# Search query generation agent for additional research
specific_field_additional_search_query_generation_agent = Agent(
    model=gemini_2o_model,
    result_type=SpecificFieldAdditionalSearchQuerySuggestion,
    system_prompt="""You are an expert at generating effective search queries for additional research.
For fields that need more detailed information, your task is to generate precise and targeted search queries.

Given a field name and a specific value, create a search query that will help find:
- More comprehensive information
- Historical context and development
- Relationships to other entities
- Technical specifications or details
- Recent developments or changes

Focus on creating queries that are:
- Specific enough to find relevant information
- Not too narrow to miss important context
- Focused on the most important aspects
- Likely to yield authoritative sources

Return a SpecificFieldAdditionalSearchQuerySuggestion with:
- field_name: The field you're researching
- field_value: The specific value you're researching
- search_query: The optimized search query
"""
)

# Search analysis agent for processing additional research
search_analysis_agent = Agent(
    model=gemini_2o_model,
    result_type=SearchAnalysisResult,
    system_prompt="""You are an expert at analyzing search results to extract comprehensive information.
Given search results for additional research on a specific field value, your task is to:
1. Extract and synthesize key information across all sources
2. Create a detailed and comprehensive description
3. Identify the most important key points
4. Suggest additional search terms for deeper research
5. List the supporting sources

Focus on:
- Factual information and specific details
- Relationships between entities and their context
- Historical development and current status
- Technical specifications and parameters
- Authoritative information from reliable sources

Organize the information logically and provide a complete picture of the topic.
Your analysis should be detailed enough to provide significant value beyond the initial information.

Return a SearchAnalysisResult with:
- detailed_description: Comprehensive synthesis of the information
- key_points: List of the most important findings
- additional_search_terms: Suggested terms for even deeper research
- supporting_sources: List of sources supporting your analysis
"""
)

async def extract_all_fields_direct(selected_sources, status_container=None, 
                                   tracker=None, company_name=None):
    """
    Extract field-specific data with improved UI organization using tabs instead of nested expanders.
    
    Args:
        selected_sources: List of selected source dictionaries
        status_container: Optional Streamlit container for displaying progress
        tracker: Optional BatchProcessTracker for tracking extraction
        company_name: Optional company name for tracking
        
    Returns:
        Dictionary of field_name -> extracted data
    """
    start_time = time.time()
    
    # Initialize result container
    field_specific_data = {}
    total_sources = len(selected_sources)
    
    # Log start of extraction if tracker provided
    if tracker and company_name:
        tracker.add_data_entry(
            company_name=company_name,
            operation="Field Extraction",
            status="Started",
            details=f"Starting extraction from {total_sources} sources"
        )
    
    # Early check for empty sources
    if not selected_sources:
        if status_container:
            status_container.warning("No sources selected for field extraction.")
        
        # Log empty sources if tracker provided
        if tracker and company_name:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Field Extraction",
                status="Skipped",
                details="No sources selected for extraction"
            )
        return {}
    
    # Create progress indicators if container provided
    progress_bar = None
    status_text = None
    if status_container:
        status_container.write("### Field Data Extraction")
        progress_bar = status_container.progress(0.0)
        status_text = status_container.empty()
    
    # Log information about selected sources
    logger.info(f"Starting field extraction for {len(selected_sources)} sources")
    if status_text:
        status_text.write(f"Starting field extraction for {len(selected_sources)} sources")
    
    # Identify target company name from sources
    target_company_name = None
    for source in selected_sources:
        if isinstance(source, dict):
            if "extracted_company_name" in source:
                target_company_name = source["extracted_company_name"]
                break
    
    # Use provided company name if available, otherwise use detected name
    tracking_company_name = company_name or target_company_name
    
    # Add target company name to sources for content validation
    for source in selected_sources:
        if isinstance(source, dict) and target_company_name:
            source["target_company_name"] = target_company_name
    
    # Display source preview if we have a status container - USING TABS INSTEAD OF EXPANDERS
    if status_container and selected_sources:
        with status_container:
            preview_tab, progress_tab = st.tabs([
                "Source Preview", "Extraction Progress"
            ])
            
            with preview_tab:
                # Create a source preview table
                preview_data = []
                for i, source in enumerate(selected_sources[:5]):  # Show first 5 for preview
                    if isinstance(source, dict):
                        if "search_result_metadata" in source:
                            preview_data.append({
                                "Source": i+1,
                                "Title": source['search_result_metadata'].get('title', 'Untitled'),
                                "URL": source['search_result_metadata'].get('url', 'No URL'),
                                "Search Query": source['search_result_metadata'].get('search_query', 'Unknown')
                            })
                        elif "company_data" in source:
                            preview_data.append({
                                "Source": i+1,
                                "Title": "Company Data",
                                "URL": "N/A",
                                "Search Query": "N/A"
                            })
                
                if preview_data:
                    st.dataframe(pd.DataFrame(preview_data))
                    if len(selected_sources) > 5:
                        st.info(f"...and {len(selected_sources) - 5} more sources")
                else:
                    st.info("No source preview available")
            
            # The progress_tab will be updated during processing
            with progress_tab:
                # Create placeholders for dynamic updates
                field_count_placeholder = st.empty()
                field_progress_placeholder = st.empty()
    
    # Process all sources in parallel with batching for better rate limiting
    batch_size = 5  # Process 5 sources at a time
    all_results = []
    filtered_sources = []
    filtered_reasons = []
    filtered_fields = 0
    filtered_field_data = []
    
    for i in range(0, len(selected_sources), batch_size):
        batch = selected_sources[i:i+batch_size]
        if status_text:
            status_text.write(f"Processing batch {i//batch_size + 1}/{math.ceil(len(selected_sources)/batch_size)}...")
        
        # Log batch processing if tracker provided
        if tracker and tracking_company_name:
            tracker.add_data_entry(
                company_name=tracking_company_name,
                operation="Process Source Batch",
                status="In Progress",
                details=f"Processing batch {i//batch_size + 1}/{math.ceil(len(selected_sources)/batch_size)}"
            )
        
        # Process batch
        source_tasks = [process_source_all_fields(source) for source in batch]
        batch_results = await asyncio.gather(*source_tasks, return_exceptions=True)
        
        # Handle batch results
        for j, result in enumerate(batch_results):
            source_idx = i + j
            source = batch[j]
            source_url = source.get('search_result_metadata', {}).get('url', 'Unknown URL') if isinstance(source, dict) else 'Unknown URL'
            
            if isinstance(result, Exception):
                logger.error(f"Error processing source {source_idx+1}: {str(result)}")
                if status_text:
                    status_text.write(f"⚠️ Error processing source {source_idx+1}: {str(result)}")
                
                # Log error if tracker provided
                if tracker and tracking_company_name:
                    tracker.add_data_entry(
                        company_name=tracking_company_name,
                        operation="Process Source",
                        status="Error",
                        details=f"Error processing source: {str(result)}",
                        url=source_url
                    )
                continue
            
            # Check if result was filtered due to invalid content
            if isinstance(result, dict) and "_validation_result" in result:
                filtered_sources += 1
                source_url = result["_validation_result"].get("url", "Unknown URL")
                content_type = result["_validation_result"].get("content_type", "Unknown")
                reason = result["_validation_result"].get("reason", "No reason provided")
                filtered_reasons.append(f"{source_url}: {content_type} - {reason}")
                
                # Log content validation if tracker provided
                if tracker and tracking_company_name:
                    tracker.log_content_validation(
                        company_name=tracking_company_name,
                        url=source_url,
                        validation_result=result["_validation_result"],
                        content_preview=source.get('search_result_metadata', {}).get('content', '')[:100] if isinstance(source, dict) else ''
                    )
                continue
            
            # Check for invalid fields
            if isinstance(result, dict) and "_invalid_fields" in result:
                invalid_fields = result.pop("_invalid_fields")  # Remove from result to avoid processing later
                source_url = "Unknown URL"
                
                # Try to get source URL
                for field_name, field_items in result.items():
                    if field_items and isinstance(field_items, list) and field_items[0].get('source'):
                        source_url = field_items[0]['source']
                        break
                
                domain = extract_domain(source_url)
                
                # Add each invalid field to our tracking
                for field_name, field_info in invalid_fields.items():
                    filtered_fields += 1
                    filtered_field_data.append({
                        "source": source_url,
                        "domain": domain,
                        "field": field_name,
                        "reason": field_info.get('reason', 'No reason provided'),
                        "confidence": field_info.get('confidence', 'Unknown')
                    })
                    
                    # Log field validation if tracker provided
                    if tracker and tracking_company_name:
                        tracker.add_data_entry(
                            company_name=tracking_company_name,
                            operation="Field Validation",
                            status="Invalid Field",
                            details=f"Invalid field {field_name}: {field_info.get('reason', 'No reason provided')}",
                            url=source_url,
                            validation_result=f"Invalid ({field_info.get('confidence', 'Unknown')})"
                        )
            
            # Only add non-empty results
            if result:
                all_results.append(result)
                
                # Log successful source processing if tracker provided
                if tracker and tracking_company_name:
                    field_count = sum(len(field_items) for field_name, field_items in result.items())
                    tracker.add_data_entry(
                        company_name=tracking_company_name,
                        operation="Process Source",
                        status="Success",
                        details=f"Extracted {field_count} field items from source",
                        url=source_url
                    )
            
            # Update progress
            if progress_bar:
                progress_bar.progress((source_idx + 1) / total_sources)
            
            # Update dynamic field count in progress tab
            if status_container and 'progress_tab' in locals():
                with progress_tab:
                    current_field_count = sum(len(field_items) for result_dict in all_results for field_name, field_items in result_dict.items())
                    field_count_placeholder.write(f"Extracted {current_field_count} field items so far")
    
    # Process all valid results
    for result in all_results:
        for field_name, field_items in result.items():
            if field_name not in field_specific_data:
                field_specific_data[field_name] = []
            field_specific_data[field_name].extend(field_items)
    
    # Complete progress for extraction phase
    if progress_bar:
        progress_bar.progress(1.0)
    
    # Log extraction phase completion if tracker provided
    if tracker and tracking_company_name:
        field_counts = {field: len(items) for field, items in field_specific_data.items()}
        tracker.add_data_entry(
            company_name=tracking_company_name,
            operation="Source Extraction",
            status="Completed",
            details=f"Extracted data for {len(field_specific_data)} field types, filtered {filtered_sources} sources and {filtered_fields} fields"
        )
    
    # Add validation summary instead of detailed logs
    if status_container:
        # Create a new validation container
        validation_container = status_container.container()
        add_validation_summary(validation_container, filtered_reasons, filtered_field_data)

    
    # ------------------ FIELD PROCESSING ------------------
    # Process fields according to their type (single-truth vs multi-source)
    processed_field_data = {}
    
    # Create new progress indicators for field processing
    field_progress = None
    field_status = None
    if status_container:
        status_container.write("### Field Processing")
        field_progress = status_container.progress(0.0)
        field_status = status_container.empty()
        
        if field_status:
            field_status.write("Starting field processing...")
    
    # Log field processing start if tracker provided
    if tracker and tracking_company_name:
        tracker.add_data_entry(
            company_name=tracking_company_name,
            operation="Field Processing",
            status="Started",
            details=f"Processing {len(field_specific_data)} field types"
        )
    
    # Create processing tasks for each field type
    field_processing_tasks = []
    field_names = []
    
    for field_name, field_items in field_specific_data.items():
        if not field_items:
            continue
            
        field_names.append(field_name)
        
        # Log field processing start if tracker provided
        if tracker and tracking_company_name:
            tracker.add_data_entry(
                company_name=tracking_company_name,
                operation=f"Process Field",
                status="Started",
                details=f"Processing field: {field_name} with {len(field_items)} items"
            )
        
        if field_name in SINGLE_TRUTH_FIELDS:
            # Single truth field processing
            field_processing_tasks.append(process_single_truth_field(field_name, field_items))
            if field_status:
                field_status.write(f"Processing single-truth field: {field_name}")
        elif field_name in MULTI_SOURCE_RESEARCH_FIELDS:
            # Multi-source field processing (no additional research by default)
            do_research = len(field_items) > 0 and field_name in ["Products / Services", "Investors"]
            field_processing_tasks.append(
                process_multi_source_field(
                    field_name, field_items, 
                    company_name=target_company_name,
                    do_additional_research=do_research,
                    tracker=tracker, 
                    tracking_company_name=tracking_company_name
                )
            )
            if field_status:
                field_status.write(f"Processing multi-source field: {field_name}")
        else:
            # Default field processing - just collect the data
            processed_data = {
                'field_name': field_name,
                'values': [],
                'sources': []
            }
            
            # Extract values and sources
            for item in field_items:
                data = item.get('data')
                source = item.get('source', 'Unknown')
                
                if data is not None:
                    value_str = str(data)
                    if hasattr(data, 'model_dump'):
                        data_dict = data.model_dump()
                        # Find the most relevant field
                        for key in data_dict:
                            if data_dict[key] and not key.startswith('_'):
                                value_str = str(data_dict[key])
                                break
                    
                    if value_str not in processed_data['values']:
                        processed_data['values'].append(value_str)
                    
                    if source not in processed_data['sources']:
                        processed_data['sources'].append(source)
                    
                    # Log extraction if tracker provided
                    if tracker and tracking_company_name:
                        tracker.log_extraction(
                            company_name=tracking_company_name,
                            field_name=field_name,
                            extracted_value=data,
                            success=True,
                            source_url=source
                        )
            
            # Add placeholder task
            field_processing_tasks.append(asyncio.sleep(0))  # No-op task
            
            # Store pre-processed result
            processed_field_data[field_name] = processed_data
    
    # Execute all field processing tasks in parallel
    if field_processing_tasks:
        field_results = await asyncio.gather(*field_processing_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(field_results):
            if i >= len(field_names):
                continue
                
            field_name = field_names[i]
            
            # Update progress
            if field_progress:
                field_progress.progress((i + 1) / len(field_names))
            
            if isinstance(result, Exception):
                logger.error(f"Error processing field {field_name}: {str(result)}")
                # Log error if tracker provided
                if tracker and tracking_company_name:
                    tracker.add_data_entry(
                        company_name=tracking_company_name,
                        operation=f"Process Field",
                        status="Error",
                        details=f"Error processing field {field_name}: {str(result)}"
                    )
                # Skip this field if there was an error
                continue
            elif result is not None:
                processed_field_data[field_name] = result
                # Log field processing success if tracker provided
                if tracker and tracking_company_name:
                    values_count = len(result.get('values', [])) if isinstance(result, dict) and 'values' in result else 0
                    sources_count = len(result.get('sources', [])) if isinstance(result, dict) and 'sources' in result else 0
                    
                    # For single truth fields, also track the conclusion
                    if field_name in SINGLE_TRUTH_FIELDS and isinstance(result, dict) and 'conclusion' in result:
                        tracker.add_data_entry(
                            company_name=tracking_company_name,
                            operation=f"Process Field",
                            status="Completed",
                            details=f"Processed single-truth field {field_name}: {result['conclusion']} (Confidence: {result.get('confidence', 'Unknown')})"
                        )
                    else:
                        tracker.add_data_entry(
                            company_name=tracking_company_name,
                            operation=f"Process Field",
                            status="Completed",
                            details=f"Processed field {field_name}: {values_count} values from {sources_count} sources"
                        )
    
    # Add any missing fields from original extraction
    for field_name, field_items in field_specific_data.items():
        if field_name not in processed_field_data:
            # Add simple aggregation
            values = []
            sources = []
            
            for item in field_items:
                data = item.get('data')
                source = item.get('source', 'Unknown')
                
                if data is not None:
                    values.append(str(data))
                    
                if source not in sources:
                    sources.append(source)
            
            processed_field_data[field_name] = {
                'field_name': field_name,
                'values': values,
                'sources': sources,
                'confidence': 'Low'
            }
    
    # Complete progress for field processing
    if field_progress:
        field_progress.progress(1.0)
    
    end_time = time.time()
    extraction_time = end_time - start_time
    
    # Generate detailed summary
    field_summary = {}
    for field_name, data in processed_field_data.items():
        field_summary[field_name] = {
            'type': 'single-truth' if field_name in SINGLE_TRUTH_FIELDS else 
                   'multi-source' if field_name in MULTI_SOURCE_RESEARCH_FIELDS else 'other',
            'sources': len(data.get('sources', [])) if isinstance(data, dict) else 0,
            'confidence': data.get('confidence', 'Unknown') if isinstance(data, dict) else 'Unknown'
        }
    
    logger.info(f"Field processing summary: {field_summary}")
    
    # Log field processing completion if tracker provided
    if tracker and tracking_company_name:
        tracker.add_data_entry(
            company_name=tracking_company_name,
            operation="Field Processing",
            status="Completed",
            details=f"Processed {len(processed_field_data)} fields in {extraction_time:.2f} seconds",
            duration=extraction_time
        )
    
    if status_container:
        # Show field processing summary - USING TABS INSTEAD OF NESTED EXPANDERS
        with status_container:
            if processed_field_data:
                summary_tab, validation_tab, details_tab = st.tabs([
                    "Field Summary", "Validation Results", "Technical Details"
                ])
                
                with summary_tab:
                    # Create a summary table
                    summary_data = []
                    for field_name, data in processed_field_data.items():
                        if not isinstance(data, dict):
                            continue
                            
                        if field_name in SINGLE_TRUTH_FIELDS and 'conclusion' in data:
                            summary_data.append({
                                "Field": field_name,
                                "Type": "Single Truth",
                                "Value": data['conclusion'],
                                "Sources": len(data.get('sources', [])),
                                "Confidence": data.get('confidence', 'Unknown')
                            })
                        elif 'values' in data and data['values']:
                            values_str = ", ".join(str(v) for v in data['values'][:3])
                            if len(data['values']) > 3:
                                values_str += f" and {len(data['values']) - 3} more"
                                
                            summary_data.append({
                                "Field": field_name,
                                "Type": "Multi Source" if field_name in MULTI_SOURCE_RESEARCH_FIELDS else "Other",
                                "Value": values_str,
                                "Sources": len(data.get('sources', [])),
                                "Confidence": data.get('confidence', 'Unknown')
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df)
                    else:
                        st.info("No summary data available.")
                
                with validation_tab:
                    # Display validation results in tabs
                    content_tab, field_tab = st.tabs(["Content Validation", "Field Validation"])
                    
                    with content_tab:
                        if filtered_sources > 0:
                            st.info(f"Filtered out {filtered_sources} sources with irrelevant content:")
                            
                            # Create dataframe for better display
                            filtered_reasons_data = []
                            for reason in filtered_reasons:
                                parts = reason.split(": ", 1)
                                if len(parts) == 2:
                                    url = parts[0]
                                    reason_details = parts[1]
                                    
                                    # Split reason details
                                    reason_parts = reason_details.split(" - ", 1)
                                    if len(reason_parts) == 2:
                                        content_type = reason_parts[0]
                                        reason_text = reason_parts[1]
                                        filtered_reasons_data.append({
                                            "Source URL": url,
                                            "Content Type": content_type,
                                            "Reason": reason_text
                                        })
                                    else:
                                        filtered_reasons_data.append({
                                            "Source URL": url,
                                            "Content Type": "Unknown",
                                            "Reason": reason_details
                                        })
                                else:
                                    filtered_reasons_data.append({
                                        "Source URL": "Unknown",
                                        "Content Type": "Unknown",
                                        "Reason": reason
                                    })
                            
                            if filtered_reasons_data:
                                st.dataframe(pd.DataFrame(filtered_reasons_data))
                            else:
                                st.write("Filtering details not available.")
                        else:
                            st.info("No sources were filtered due to content validation.")
                    
                    with field_tab:
                        if filtered_fields > 0:
                            st.info(f"Filtered out {filtered_fields} irrelevant fields:")
                            
                            # Group by domain for better display
                            domains = {}
                            for field_data in filtered_field_data:
                                domain = field_data["domain"]
                                if domain not in domains:
                                    domains[domain] = []
                                domains[domain].append(field_data)
                            
                            # Display a summary table
                            filtered_fields_data = []
                            for domain, fields in domains.items():
                                for field_data in fields:
                                    filtered_fields_data.append({
                                        "Domain": domain,
                                        "Field": field_data['field'],
                                        "Reason": field_data['reason'],
                                        "Confidence": field_data['confidence']
                                    })
                            
                            if filtered_fields_data:
                                st.dataframe(pd.DataFrame(filtered_fields_data))
                            else:
                                st.write("Field filtering details not available.")
                        else:
                            st.info("No fields were filtered due to field validation.")
                
                with details_tab:
                    st.json(field_summary)
            
            st.success(f"Completed extraction and processing in {extraction_time:.2f} seconds")
    
    return processed_field_data

def extract_content_from_source(source):
    """
    Extract content, URL and source type from a source with improved extraction logic.
    
    Args:
        source: Source dictionary
        
    Returns:
        Tuple of (content, url, source_type)
    """
    content = None
    url = "Unknown"
    source_type = "Unknown"
    
    if not isinstance(source, dict):
        return None, url, source_type
    
    # Path 1: search_result_metadata
    if "search_result_metadata" in source:
        metadata = source["search_result_metadata"]
        if isinstance(metadata, dict):
            # Try raw_content first as it's often more complete
            raw_content = metadata.get("raw_content", "")
            content = metadata.get("content", "")
            
            # Use raw_content if available and content is empty or too short
            if raw_content and (not content or len(content) < len(raw_content)):
                content = raw_content
                source_type = "raw_content"
            else:
                source_type = "content"
                
            url = metadata.get("url", "Unknown")
    
    # Path 2: direct content
    elif "content" in source:
        content = source["content"]
        source_type = "direct_content"
        if "url" in source:
            url = source["url"]
    
    # Path 3: company_data fields
    elif "company_data" in source and isinstance(source["company_data"], dict):
        company_data = source["company_data"]
        
        # Try different fields that might contain useful content
        content_fields = ["description_abstract", "product_type", "scientific_domain"]
        for field in content_fields:
            if field in company_data and company_data[field]:
                if content:
                    content += "\n\n" + str(company_data[field])
                else:
                    content = str(company_data[field])
                source_type = f"company_data_{field}"
        
        # Also try to concatenate list fields
        list_fields = ["product_name", "company_url", "hq_locations", "relevant_segments", 
                       "investor_name", "competitors"]
        for field in list_fields:
            if field in company_data and company_data[field]:
                field_content = ""
                if isinstance(company_data[field], list):
                    field_content = ", ".join(str(item) for item in company_data[field] if item)
                else:
                    field_content = str(company_data[field])
                
                if field_content:
                    if content:
                        content += f"\n\n{field}: {field_content}"
                    else:
                        content = f"{field}: {field_content}"
                    source_type = f"company_data_{field}"
    
    return content, url, source_type

def format_field_for_validation(extraction_result, field_name):
    """
    Format field value for validation with appropriate formatting for different types.
    
    Args:
        extraction_result: The extracted field data
        field_name: Name of the field
        
    Returns:
        Formatted string representation of the field value
    """
    try:
        # For Pydantic models, use model_dump_json for compact representation
        if hasattr(extraction_result, 'model_dump_json'):
            return extraction_result.model_dump_json()
        
        # For Pydantic models with dict method
        elif hasattr(extraction_result, 'dict'):
            return json.dumps(extraction_result.dict())
        
        # For dictionaries
        elif isinstance(extraction_result, dict):
            return json.dumps(extraction_result)
        
        # For lists
        elif isinstance(extraction_result, list):
            # Short lists can be shown directly
            if len(extraction_result) <= 5:
                return json.dumps(extraction_result)
            # For longer lists, show summary
            else:
                return f"List with {len(extraction_result)} items. First 3: {json.dumps(extraction_result[:3])}..."
        
        # For other types
        else:
            return str(extraction_result)
            
    except Exception as e:
        # Fallback to basic string representation
        return f"[Error formatting: {str(e)}] {str(extraction_result)}"

async def process_source_all_fields(source):
    """
    Process a single source for all fields in parallel with enhanced error handling.
    
    Args:
        source: Source dictionary
        
    Returns:
        Dictionary of field_name -> field_data_list
    """
    # Extract content from source with improved content extraction
    content, url, source_type = extract_content_from_source(source)
    
    # If no content found or too short, return empty result
    if not content or len(content) < 10:  # Require at least 10 characters
        logger.warning(f"Insufficient content extracted from source: {url}")
        return {}

    # Get target company name if available
    target_company = ""
    if "target_company_name" in source:
        target_company = source["target_company_name"]
    elif "extracted_company_name" in source:
        target_company = source["extracted_company_name"]
    
    # Validate content relevance if target company provided
    if content is not None and target_company:
        try:
            async with asyncio.timeout(20):  # 20-second timeout
                validation_result = await content_validation_agent.run(
                    f"""
                    Target Company: {target_company}
                    URL: {url}
                    Content Preview:
                    {content[:1500] if content else "No content available"}
                    
                    Is this content valid information about {target_company} or is it an access barrier/irrelevant content?
                    """
                )
            
            # Skip further processing if content is invalid with high confidence
            if not validation_result.data.is_valid and validation_result.data.confidence in ["High", "Very High"]:
                logger.warning(f"Filtered content from {url}: {validation_result.data.content_type} - {validation_result.data.reason}")
                return {
                    "_validation_result": {
                        "is_valid": False,
                        "content_type": validation_result.data.content_type,
                        "confidence": validation_result.data.confidence,
                        "reason": validation_result.data.reason,
                        "url": url
                    }
                }
        except (asyncio.TimeoutError, Exception) as e:
            # If validation fails, log but continue processing
            logger.warning(f"Content validation failed for {url}: {str(e)}")
    
    # Create tasks for all field extractors with dynamic batching
    # Group fields into batches for optimal parallel execution
    all_fields = list(COLUMN_TO_AGENT_FUNCTION.items())
    batch_size = min(5, len(all_fields))  # Limit batch size to prevent resource exhaustion
    
    field_results = {}
    invalid_fields = {}
    
    # Process fields in batches for better resource utilization
    for i in range(0, len(all_fields), batch_size):
        batch = all_fields[i:i+batch_size]
        
        # Create extraction tasks for this batch
        extraction_tasks = []
        field_names = []
        
        for field_name, agent_function in batch:
            extraction_tasks.append(agent_function(content))
            field_names.append(field_name)
        
        # Execute all extraction tasks in parallel with timeout
        try:
            async with asyncio.timeout(30):  # 30-second timeout for batch
                batch_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            # Process results for this batch
            for j, field_name in enumerate(field_names):
                extraction_result = batch_results[j]
                
                # Handle errors gracefully
                if isinstance(extraction_result, Exception):
                    logger.error(f"Error extracting {field_name}: {str(extraction_result)}")
                    continue
                
                # Skip empty results
                if not extraction_result:
                    continue
                
                # Validate field data for relevance to target company
                if target_company:
                    try:
                        # Format field value for validation
                        field_value_str = format_field_for_validation(extraction_result, field_name)
                        
                        # Validate field with timeout
                        async with asyncio.timeout(15):  # 15-second timeout for validation
                            validation_result = await field_validation_agent.run(
                                f"""
                                Target Company: {target_company}
                                Source Domain: {extract_domain(url)}
                                Field Name: {field_name}
                                Field Value: {field_value_str}
                                Source URL: {url}
                                
                                Determine if this field value contains valid information about {target_company}
                                or if it's generic/irrelevant information (about the source rather than the target).
                                """
                            )
                        
                        # Skip fields that are invalid with high confidence
                        if not validation_result.data.is_valid and validation_result.data.confidence in ["High", "Very High"]:
                            logger.info(f"Filtered invalid field {field_name} from {url}: {validation_result.data.reason}")
                            invalid_fields[field_name] = {
                                "reason": validation_result.data.reason,
                                "confidence": validation_result.data.confidence
                            }
                            continue
                            
                    except (asyncio.TimeoutError, Exception) as e:
                        # If validation fails, continue with the field (don't filter)
                        logger.warning(f"Field validation failed for {field_name}: {str(e)}")
                
                # Add valid field data to results
                if field_name not in field_results:
                    field_results[field_name] = []
                
                field_results[field_name].append({
                    'data': extraction_result,
                    'source': url,
                    'source_type': source_type,
                    'extraction_date': datetime.now().isoformat()
                })
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout processing batch of fields for {url}")
        except Exception as e:
            logger.error(f"Error processing batch of fields for {url}: {str(e)}")
    
    # Add invalid fields info to result if any fields were filtered
    if invalid_fields:
        field_results["_invalid_fields"] = invalid_fields
    
    return field_results

def format_and_merge_data(data_dict, data, field_name):
    """Helper function to format and merge field data for display."""
    if not data:
        return
    
    if field_name == "Products / Services":
        if hasattr(data, 'product_name'):
            if "Product Name" not in data_dict:
                data_dict["Product Name"] = []
            if data.product_name not in data_dict["Product Name"]:
                data_dict["Product Name"].append(data.product_name)
        
        if hasattr(data, 'product_type'):
            data_dict["Product Type"] = data.product_type
        
        if hasattr(data, 'description_abstract'):
            data_dict["Description"] = data.description_abstract
            
    elif field_name == "Investors":
        if hasattr(data, 'investors'):
            if "Investors" not in data_dict:
                data_dict["Investors"] = []
            
            for investor in data.investors:
                if hasattr(investor, 'investor_name'):
                    if investor.investor_name not in data_dict["Investors"]:
                        data_dict["Investors"].append(investor.investor_name)
    
    elif field_name == "Business Model(s)":
        if hasattr(data, 'business_models'):
            if "Business Models" not in data_dict:
                data_dict["Business Models"] = []
            
            for model in data.business_models:
                if model not in data_dict["Business Models"]:
                    data_dict["Business Models"].append(model)
    
    elif field_name == "Year Founded":
        if hasattr(data, 'year_founded'):
            data_dict["Year Founded"] = data.year_founded
    
    elif field_name == "HQ Location":
        if hasattr(data, 'locations'):
            if "Locations" not in data_dict:
                data_dict["Locations"] = []
            
            for location in data.locations:
                if location not in data_dict["Locations"]:
                    data_dict["Locations"].append(location)
    
    elif field_name == "Advisor / Board Members / Key People":
        if hasattr(data, 'name') and hasattr(data, 'role'):
            if "Key People" not in data_dict:
                data_dict["Key People"] = []
            
            person_info = {"Name": data.name, "Role": data.role}
            
            # Check if this person is already in the list
            already_exists = False
            for person in data_dict["Key People"]:
                if person.get("Name") == data.name:
                    already_exists = True
                    break
            
            if not already_exists:
                data_dict["Key People"].append(person_info)
    else:
        # For other field types, use a generic approach
        if isinstance(data, BaseModel):
            # Convert Pydantic models to dictionaries
            data_dict.update(data.model_dump() if hasattr(data, 'model_dump') else data.dict())
        elif isinstance(data, dict):
            data_dict.update(data)
        else:
            # For other types, just store the string representation
            data_dict[field_name] = str(data)

    return data_dict

def display_aggregated_field_data(field_name, field_items):
    """Display aggregated field data in a structured format."""
    # Different display based on field type
    if field_name == "Products / Services":
        # Extract all product names
        all_products = []
        for item in field_items:
            if isinstance(item, dict) and item.get('data'):
                data = item.get('data')
                if hasattr(data, 'product_name') and data.product_name:
                    all_products.append(data.product_name)
                elif isinstance(data, dict) and 'product_name' in data:
                    all_products.append(data['product_name'])
            
        if all_products:
            # Count occurrences of each product
            from collections import Counter
            product_counts = Counter(all_products)
            
            # Create dataframe for display
            st.write("**Products mentioned across sources:**")
            st.dataframe(
                pd.DataFrame({
                    "Product": list(product_counts.keys()),
                    "Mentions": list(product_counts.values())
                }).sort_values("Mentions", ascending=False)
            )
        else:
            st.info("No product data available.")
            
    elif field_name == "Investors":
        # Extract all investors
        all_investors = []
        for item in field_items:
            if isinstance(item, dict) and item.get('data'):
                data = item.get('data')
                if hasattr(data, 'investors'):
                    for investor in data.investors:
                        if hasattr(investor, 'investor_name'):
                            all_investors.append(investor.investor_name)
                elif isinstance(data, dict) and 'investors' in data:
                    for investor in data['investors']:
                        if isinstance(investor, dict) and 'investor_name' in investor:
                            all_investors.append(investor['investor_name'])
        
        if all_investors:
            # Count occurrences of each investor
            from collections import Counter
            investor_counts = Counter(all_investors)
            
            # Create dataframe for display
            st.write("**Investors mentioned across sources:**")
            st.dataframe(
                pd.DataFrame({
                    "Investor": list(investor_counts.keys()),
                    "Mentions": list(investor_counts.values())
                }).sort_values("Mentions", ascending=False)
            )
        else:
            st.info("No investor data available.")
            
    else:
        # Generic handling for other field types
        st.write("**Aggregated data from all sources:**")
        
        # Create an aggregation dictionary
        all_data = {}
        
        for item in field_items:
            if isinstance(item, dict) and item.get('data'):
                data = item.get('data')
                if data:
                    format_and_merge_data(all_data, data, field_name)
        
        # Display the aggregated data
        if all_data:
            st.json(all_data)
        else:
            st.info("No aggregated data available.")


def create_non_exact_match_expander(idx, result, cleaned_search_query_name, entity_verification_results, selection_results):
    """Create a consistently structured expander for non-exact matches."""
    extracted_entity_name = result.get('extracted_company_name', 'Unknown')
    search_result_metadata = result.get('search_result_metadata', {})
    title = search_result_metadata.get('title', 'No title')
    url = search_result_metadata.get('url', 'No URL')
    
    # Initialize verification_info to None
    verification_info = None
    
    # Check if verification info exists
    if entity_verification_results and extracted_entity_name in entity_verification_results:
        verification_info = entity_verification_results[extracted_entity_name]
    
    # Determine if checkbox should be disabled
    disabled = False
    if verification_info is not None and not verification_info.are_same_entity:
        if verification_info.confidence_level in ["High", "Very High"]:
            disabled = True
    
    # Create checkbox key
    checkbox_key = f"non_exact_result_checkbox_{cleaned_search_query_name}_{idx}"
    
    # Set default checkbox value
    default_checkbox_value = False
    if verification_info is not None and verification_info.are_same_entity:
        if verification_info.confidence_level in ["High", "Very High"]:
            default_checkbox_value = True
    elif idx < len(selection_results) and selection_results[idx].data:
        if selection_results[idx].data.include_in_table:
            default_checkbox_value = True
    
    # Create status icon
    status_icon = ""
    if verification_info is not None:
        status_icon = "✅" if verification_info.are_same_entity else "❌"
    
    return {
        "verification_info": verification_info,
        "title": title,
        "extracted_entity_name": extracted_entity_name,
        "url": url,
        "checkbox_key": checkbox_key,
        "default_checkbox_value": default_checkbox_value,
        "disabled": disabled,
        "status_icon": status_icon
    }


# Function to display field-specific data with source attribution
async def display_field_specific_data_with_source_attribution(field_specific_data, extraction_status_container):
    """
    Display field-specific data with source-specific attributions using tabs.
    
    Args:
        field_specific_data: Dictionary mapping field names to extracted data items
        extraction_status_container: Streamlit container for status updates
    """
    st.header("Extracted Structured Information")
    
    if not field_specific_data:
        st.info("No field-specific structured data extracted from the selected sources.")
        return
    
    extraction_status_container.info("Organizing field data by source...")
    
    # Create a summary table with source-specific attributions
    structured_data_summary = {
        "Feature": [],
        "Value": [],
        "Key Points": [],
        "Sources": [],
        "Confidence": []
    }
    
    # Process each field type
    for field_name, field_items in field_specific_data.items():
        if not field_items:
            continue
        
        # Add field name to Feature column
        structured_data_summary["Feature"].append(field_name)
        
        # Get unique sources - FIX: safely handle different item types
        source_urls = []
        for item in field_items:
            if isinstance(item, dict):
                source_url = item.get('source', '')
                if source_url:
                    source_urls.append(source_url)
            elif isinstance(item, str):
                # If item is a string, just add it directly
                source_urls.append(item)
        
        source_count = len(set(source_urls))
        
        # Group findings by source
        sources_to_findings = {}
        source_confidence = {}
        
        for idx, item in enumerate(field_items):
            # FIX: Handle different item types appropriately
            if isinstance(item, dict):
                source_url = item.get('source', f'Unknown Source {idx}')
                data = item.get('data')
            else:
                # For non-dict items, use defaults
                source_url = f'Unknown Source {idx}'
                data = item
            
            if not data:
                continue
                
            # Extract the main finding from the data
            finding = extract_main_finding(field_name, data)
            
            if finding:
                if source_url not in sources_to_findings:
                    sources_to_findings[source_url] = []
                    # Assign confidence based on URL domain
                    source_confidence[source_url] = calculate_source_confidence(source_url)
                
                sources_to_findings[source_url].append(finding)
        
        # Create Value column with source-specific findings
        value_text_parts = []
        
        for idx, (source_url, findings) in enumerate(sources_to_findings.items()):
            if not findings:
                continue
                
            # Generate a summary of findings from this source
            finding_summary = "; ".join(findings[:3])
            if len(findings) > 3:
                finding_summary += f" and {len(findings) - 3} more findings"
                
            value_text_parts.append(f"Source {idx+1} ({get_domain(source_url)}): {finding_summary}")
        
        value_text = "\n\n".join(value_text_parts)
        structured_data_summary["Value"].append(value_text)
        
        # Create Key Points column with source attribution
        key_points_parts = []
        
        # Select top findings across sources (max 5)
        top_findings = []
        for source_url, findings in sources_to_findings.items():
            if findings:
                # Take the first finding from each source
                domain = get_domain(source_url)
                top_findings.append((findings[0], domain))
                
                # If more than one finding from this source, add the second one too
                if len(findings) > 1:
                    top_findings.append((findings[1], domain))
                    
                # Stop after 5 findings
                if len(top_findings) >= 5:
                    break
        
        # Format key points with source attribution
        for finding, domain in top_findings[:5]:
            key_points_parts.append(f"{finding} ({domain})")
            
        key_points_text = ", ".join(key_points_parts)
        structured_data_summary["Key Points"].append(key_points_text)
        
        # Add source count
        structured_data_summary["Sources"].append(str(source_count))
        
        # Calculate overall confidence based on source confidence and consistency
        overall_confidence = calculate_overall_confidence(sources_to_findings, source_confidence)
        structured_data_summary["Confidence"].append(overall_confidence)
    
    extraction_status_container.success(f"Organized data for {len(structured_data_summary['Feature'])} field types")
    
    # Display the summary table
    if structured_data_summary["Feature"]:
        st.subheader("Structured Information Summary")
        # Convert to DataFrame and display as table
        summary_df = pd.DataFrame(structured_data_summary)
        st.table(summary_df)
    
    # IMPROVED UI: Organize fields by category in tabs
    field_categories = {
        "Basic Information": ["Organization Type", "HQ Location", "Year Founded", "Female Co-Founder?"],
        "Products & Services": ["Products / Services", "Scientific Domain", "Business Model(s)", 
                              "Lab or Proprietary Data Generation?", "Drug Pipeline?"],
        "Business Relationships": ["Publicly Announced Partnerships", "Investors", 
                                 "Advisor / Board Members / Key People"],
        "Financial Information": ["Funding Stage", "RoM Estimated Funding or Market Cap for Public Companies"],
        "Other Information": ["Relevant Segments", "Watchlist"]
    }
    
    # Create tabs for each category
    if field_specific_data:
        st.subheader("Detailed Field Information")
        
        # Determine which categories have fields
        categories_with_fields = []
        for category, fields in field_categories.items():
            if any(field in field_specific_data for field in fields):
                categories_with_fields.append(category)
        
        if categories_with_fields:
            tabs = st.tabs(categories_with_fields)
            
            # Fill each tab with its fields
            for i, category in enumerate(categories_with_fields):
                with tabs[i]:
                    fields = field_categories[category]
                    for field_name in fields:
                        if field_name in field_specific_data and field_specific_data[field_name]:
                            # Get confidence for color coding
                            confidence = "Unknown"
                            for feature_idx, feature in enumerate(structured_data_summary["Feature"]):
                                if feature == field_name:
                                    confidence = structured_data_summary["Confidence"][feature_idx]
                                    break
                            
                            with st.expander(f"{field_name} ({len(field_specific_data[field_name])} items, Confidence: {confidence})", expanded=False):
                                # For each field, create tabs to organize sources
                                source_tabs = st.tabs(["By Source", "Aggregated View"])
                                
                                # Tab 1: Organize by source
                                with source_tabs[0]:
                                    # Group items by source
                                    items_by_source = {}
                                    for item in field_specific_data[field_name]:
                                        # FIX: Handle different item types
                                        if isinstance(item, dict):
                                            source_url = item.get('source', 'Unknown Source')
                                            if source_url not in items_by_source:
                                                items_by_source[source_url] = []
                                            items_by_source[source_url].append(item)
                                        else:
                                            # For non-dict items
                                            source_url = 'Unknown Source'
                                            if source_url not in items_by_source:
                                                items_by_source[source_url] = []
                                            items_by_source[source_url].append({"data": item, "source": source_url})
                                    
                                    # Display each source and its findings
                                    for source_url, items in items_by_source.items():
                                        domain = get_domain(source_url)
                                        st.markdown(f"**Source: [{domain}]({source_url})**")
                                        
                                        for item in items:
                                            data = item.get('data') if isinstance(item, dict) else item
                                            if data:
                                                # Display formatted data based on field type
                                                display_field_data(field_name, data)
                                        
                                        st.divider()
                                
                                # Tab 2: Aggregated view
                                with source_tabs[1]:
                                    # Display aggregated information
                                    display_aggregated_field_data(field_name, field_specific_data[field_name])

def extract_main_finding(field_name, data):
    """
    Extract the main finding from field data based on field type with enhanced extraction.
    
    Args:
        field_name: Name of the field
        data: Field data (could be Pydantic model or dict)
        
    Returns:
        String containing the main finding
    """
    try:
        # Enhanced ProductServiceDetails extraction
        if field_name == "Products / Services":
            if hasattr(data, 'product_name') and data.product_name:
                product_type = getattr(data, 'product_type', '')
                return f"{data.product_name} ({product_type})" if product_type else data.product_name
            elif hasattr(data, 'product_type'):
                return data.product_type
            elif isinstance(data, dict):
                if 'product_name' in data and data['product_name']:
                    product_type = data.get('product_type', '')
                    return f"{data['product_name']} ({product_type})" if product_type else data['product_name']
                elif 'product_type' in data:
                    return data['product_type']
        
        # Enhanced Business Model extraction
        elif field_name == "Business Model(s)":
            if hasattr(data, 'primary_models') and data.primary_models:
                return ", ".join(data.primary_models[:3])
            elif isinstance(data, dict) and 'primary_models' in data:
                models = data['primary_models']
                if isinstance(models, list) and models:
                    return ", ".join(models[:3])
            elif hasattr(data, 'business_models') and data.business_models:
                return ", ".join(data.business_models[:3])
            elif isinstance(data, dict) and 'business_models' in data:
                models = data['business_models']
                if isinstance(models, list) and models:
                    return ", ".join(models[:3])
        
        # Enhanced Scientific Domain extraction
        elif field_name == "Scientific Domain":
            if hasattr(data, 'primary_domain'):
                sub_domains = getattr(data, 'sub_domains', [])
                if sub_domains:
                    return f"{data.primary_domain} ({', '.join(sub_domains[:2])})"
                return data.primary_domain
            elif isinstance(data, dict) and 'primary_domain' in data:
                sub_domains = data.get('sub_domains', [])
                if sub_domains:
                    return f"{data['primary_domain']} ({', '.join(sub_domains[:2])})"
                return data['primary_domain']
        
        # Enhanced Key People extraction
        elif field_name == "Advisor / Board Members / Key People":
            if hasattr(data, 'name') and hasattr(data, 'role'):
                # Check for enhanced model with is_founder etc.
                if hasattr(data, 'is_founder') and data.is_founder:
                    return f"{data.name} (Founder, {data.role})"
                elif hasattr(data, 'is_executive') and data.is_executive:
                    return f"{data.name} (Executive, {data.role})"
                elif hasattr(data, 'is_board_member') and data.is_board_member:
                    return f"{data.name} (Board Member, {data.role})"
                else:
                    return f"{data.name} ({data.role})"
            elif isinstance(data, dict):
                if 'name' in data and 'role' in data:
                    if data.get('is_founder', False):
                        return f"{data['name']} (Founder, {data['role']})"
                    elif data.get('is_executive', False):
                        return f"{data['name']} (Executive, {data['role']})"
                    elif data.get('is_board_member', False):
                        return f"{data['name']} (Board Member, {data['role']})"
                    else:
                        return f"{data['name']} ({data['role']})"
        
        # Enhanced Competitive Landscape extraction
        elif field_name == "Competitive Landscape":
            if hasattr(data, 'direct_competitors') and data.direct_competitors:
                competitors = [comp.company_name for comp in data.direct_competitors[:3]]
                return f"Competitors: {', '.join(competitors)}"
            elif isinstance(data, dict) and 'direct_competitors' in data:
                competitors = data['direct_competitors']
                if isinstance(competitors, list) and competitors:
                    comp_names = []
                    for comp in competitors[:3]:
                        if isinstance(comp, dict) and 'company_name' in comp:
                            comp_names.append(comp['company_name'])
                    if comp_names:
                        return f"Competitors: {', '.join(comp_names)}"
        
        # Original handler for other field types would go here
        # For example, the Funding Rounds and Investors fields
        elif field_name == "Funding Rounds":
            if hasattr(data, 'funding_rounds'):
                for round_info in data.funding_rounds[:1]:  # Just show the first round for summary
                    formatted_value = ""
                    if round_info.round_date:
                        formatted_value += f"{round_info.round_date}"
                    if round_info.amount:
                        formatted_value += f" - {round_info.amount}"
                    if round_info.round_type:
                        formatted_value += f" - {round_info.round_type} Round"
                    return formatted_value
            elif hasattr(data, 'total_funding') and data.total_funding:
                return f"Total Funding: {data.total_funding}"
                
        elif field_name == "Investors":
            if hasattr(data, 'investors') and data.investors:
                investor_names = []
                for investor in data.investors[:3]:  # Show top 3 investors
                    if hasattr(investor, 'investor_name'):
                        investor_names.append(investor.investor_name)
                if investor_names:
                    return f"Investors: {', '.join(investor_names)}"
            
        elif field_name == "Publicly Announced Partnerships":
            partnership_info = []
            if hasattr(data, 'company_name') and data.company_name:
                partnership_info.append(data.company_name)
            if hasattr(data, 'partnership_entities') and data.partnership_entities:
                partnership_info.append(data.partnership_entities)
            return " partnership with ".join(partnership_info) if partnership_info else "Partnership details found"
        
    except Exception as e:
        logger.error(f"Error extracting main finding for {field_name}: {e}")
    
    # Generic fallback for any field type if specific handling failed
    if isinstance(data, BaseModel):
        model_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        first_value = next((v for k, v in model_dict.items() if v and not k.startswith('_')), None)
        return str(first_value)[:100] if first_value else f"{field_name} data found"
    elif isinstance(data, dict):
        first_value = next((v for k, v in data.items() if v and not k.startswith('_')), None)
        return str(first_value)[:100] if first_value else f"{field_name} data found"
    else:
        return str(data)[:100]

def display_field_data(field_name, data):
    """
    Display field data in a formatted way based on field type.
    
    Args:
        field_name: Name of the field
        data: Field data (could be Pydantic model or dict)
    """
    if field_name == "Products / Services":
        if hasattr(data, 'product_name'):
            st.write(f"**Product Name:** {data.product_name}")
        if hasattr(data, 'product_type'):
            st.write(f"**Product Type:** {data.product_type}")
        if hasattr(data, 'description_abstract'):
            st.write(f"**Description:** {data.description_abstract}")
    elif isinstance(data, BaseModel):
        # For other Pydantic models, display all non-empty fields
        model_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        for key, value in model_dict.items():
            if value and not key.startswith('_'):
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    elif isinstance(data, dict):
        # For dictionaries, display all non-empty fields
        for key, value in data.items():
            if value and not key.startswith('_'):
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    else:
        # Just display as string for other types
        st.write(data)

def get_domain(url):
    """
    Extract the domain name from a URL.
    
    Args:
        url: URL string
        
    Returns:
        Domain name
    """
    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # If domain is empty, take the first part of the path
        if not domain and parsed_url.path:
            domain = parsed_url.path.split('/')[0]
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain or "Unknown source"
    except:
        return "Unknown source"

def calculate_source_confidence(url):
    """
    Calculate a confidence score for a source based on its domain.
    
    Args:
        url: Source URL
        
    Returns:
        Confidence level string: "Very High", "High", "Medium", or "Low"
    """
    domain = get_domain(url).lower()
    
    # High authority domains
    high_authority_domains = [
        'crunchbase.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'reuters.com',
        'forbes.com', 'techcrunch.com', 'sec.gov', 'fda.gov', 'nih.gov',
        'pitchbook.com', 'nature.com', 'science.org', 'ycombinator.com'
    ]
    
    # Medium authority domains
    medium_authority_domains = [
        'linkedin.com', 'medium.com', 'entrepreneur.com', 'inc.com',
        'businessinsider.com', 'venturebeat.com', 'wired.com', 'fastcompany.com'
    ]
    
    if any(domain.endswith(high_domain) or domain == high_domain for high_domain in high_authority_domains):
        return "Very High"
    elif any(domain.endswith(med_domain) or domain == med_domain for med_domain in medium_authority_domains):
        return "High"
    elif domain.endswith('.edu') or domain.endswith('.gov') or domain.endswith('.org'):
        return "High"
    else:
        return "Medium"

def calculate_overall_confidence(sources_to_findings, source_confidence):
    """
    Calculate overall confidence for a field based on source confidence and consistency.
    
    Args:
        sources_to_findings: Dictionary mapping sources to their findings
        source_confidence: Dictionary mapping sources to their confidence levels
        
    Returns:
        Overall confidence level: "Very High", "High", "Medium", or "Low"
    """
    if not sources_to_findings:
        return "Low"
    
    # Count sources by confidence level
    confidence_counts = {"Very High": 0, "High": 0, "Medium": 0, "Low": 0}
    for source in sources_to_findings:
        conf = source_confidence.get(source, "Low")
        confidence_counts[conf] += 1
    
    # Calculate a weighted score
    total_sources = len(sources_to_findings)
    weighted_score = (
        confidence_counts["Very High"] * 4 +
        confidence_counts["High"] * 3 +
        confidence_counts["Medium"] * 2 +
        confidence_counts["Low"] * 1
    ) / total_sources if total_sources > 0 else 0
    
    # Check for consistency across sources
    all_findings = []
    for findings in sources_to_findings.values():
        all_findings.extend(findings)
    
    # Simple consistency check - are there many different findings or mostly the same?
    unique_findings = len(set(all_findings))
    consistency_factor = 1.0 if unique_findings <= 2 else (
        0.9 if unique_findings <= 4 else (
            0.8 if unique_findings <= 6 else 0.7
        )
    )
    
    final_score = weighted_score * consistency_factor
    
    # Map score to confidence level
    if final_score >= 3.5:
        return "Very High"
    elif final_score >= 2.5:
        return "High"
    elif final_score >= 1.5:
        return "Medium"
    else:
        return "Low"

# 1. Define the Pydantic model for field summaries
class FieldSummary(BaseModel):
    """AI-generated summary for a specific field type."""
    field_name: str = Field(..., description="Name of the field being summarized.")
    summary: str = Field(..., description="Concise summary of the field data.")
    key_items: List[str] = Field(default_factory=list, description="Key items or highlights from the field data.")
    source_count: int = Field(..., description="Number of unique sources this data comes from.")
    confidence: str = Field(..., description="Confidence level: Low, Medium, High, Very High")

# 2. Create the Gemini agent for field summarization
field_summary_agent = Agent(
    model=gemini_2o_model,
    result_type=FieldSummary,
    system_prompt="""You are an expert at concisely summarizing structured field data about companies.

Given a collection of field-specific items of the same type, create a clear and concise summary that:
1. Highlights the most important information
2. Identifies consistent patterns across multiple sources
3. Notes any significant variations or contradictions
4. Prioritizes information from reputable sources

Provide your summary in the FieldSummary format with:
- A concise 1-2 sentence summary of the field data
- Key items or highlights (limited to 3-5)
- Source count (calculated from the data)
- Confidence level based on consistency and source quality

Your summary should be suitable for a business intelligence dashboard."""
)

# 3. Function to generate field summaries using the agent
async def generate_field_summaries(field_specific_data):
    """
    Generate AI summaries for each field type in the extracted data.
    
    Args:
        field_specific_data: Dictionary mapping field names to extracted data items
        
    Returns:
        Dictionary mapping field names to FieldSummary objects
    """
    field_summaries = {}
    summary_tasks = []
    field_names = []
    
    # Process each field type
    for field_name, field_items in field_specific_data.items():
        if not field_items:
            continue
            
        # Calculate source count
        source_count = len(set(item.get('source', '') for item in field_items))
        
        # Prepare data for the agent
        field_data = {
            "field_name": field_name,
            "items": [],
            "sources": [item.get('source', '') for item in field_items],
            "source_count": source_count
        }
        
        # Process different field types appropriately
        for item in field_items:
            if not item.get('data'):
                continue
                
            data = item['data']
            if isinstance(data, BaseModel):
                # Convert Pydantic models to dictionaries
                data_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
                field_data["items"].append(data_dict)
            elif isinstance(data, dict):
                field_data["items"].append(data)
            else:
                field_data["items"].append(str(data))
        
        # Only create summary tasks for fields with items
        if field_data["items"]:
            field_names.append(field_name)
            summary_tasks.append(field_summary_agent.run(
                user_prompt=f"""
                Summarize the following {field_name} data for a company:
                
                Field data: {json.dumps(field_data, default=str)}
                
                Create a concise summary highlighting the most important information about this {field_name} data.
                Identify 3-5 key items or highlights that would be most relevant for business intelligence.
                """
            ))
    
    # Execute all summary tasks in parallel with rate limiting
    if summary_tasks:
        # Use semaphore for rate limiting
        sem = asyncio.Semaphore(5)  # Limit to 5 concurrent API calls
        
        async def run_with_semaphore(task):
            async with sem:
                return await task
                
        wrapped_tasks = [run_with_semaphore(task) for task in summary_tasks]
        summary_results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
        
        # Process summary results
        for i, field_name in enumerate(field_names):
            result = summary_results[i]
            if not isinstance(result, Exception) and result and hasattr(result, 'data'):
                field_summaries[field_name] = result.data
            else:
                # Create a basic summary if agent fails
                source_count = len(set(item.get('source', '') for item in field_specific_data[field_name]))
                field_summaries[field_name] = FieldSummary(
                    field_name=field_name,
                    summary=f"{len(field_specific_data[field_name])} {field_name} items found",
                    key_items=[],
                    source_count=source_count,
                    confidence="Low"
                )
    
    return field_summaries

# 4. Function to display field data with AI-generated summaries
async def display_field_specific_data_with_summaries(field_specific_data, extraction_status_container):
    """
    Display field-specific data with AI-generated summaries and detailed expandable sections.
    
    Args:
        field_specific_data: Dictionary mapping field names to extracted data items
        extraction_status_container: Streamlit container for status updates
    """
    st.header("Extracted Structured Information")
    
    if not field_specific_data:
        st.info("No field-specific structured data extracted from the selected sources.")
        return
    
    # Generate AI summaries for all fields
    extraction_status_container.info("Generating AI summaries for field data...")
    field_summaries = await generate_field_summaries(field_specific_data)
    extraction_status_container.success(f"Generated summaries for {len(field_summaries)} field types")
    
    # Create a summary table using the AI-generated summaries
    structured_data_summary = {
        "Feature": [],
        "Value": [],
        "Key Points": [],
        "Sources": [],
        "Confidence": []
    }
    
    # Add summary data for each field
    for field_name, summary in field_summaries.items():
        structured_data_summary["Feature"].append(field_name)
        structured_data_summary["Value"].append(summary.summary)
        structured_data_summary["Key Points"].append(", ".join(summary.key_items[:3]) if summary.key_items else "")
        structured_data_summary["Sources"].append(f"{summary.source_count}")
        structured_data_summary["Confidence"].append(summary.confidence)
    
    # Display the summary table
    if structured_data_summary["Feature"]:
        st.subheader("Structured Information Summary")
        # Convert to DataFrame and display as table for consistent formatting with company overview
        summary_df = pd.DataFrame(structured_data_summary)
        st.table(summary_df)
    
    # Display detailed expandable sections for each field
    for field_name, field_items in field_specific_data.items():
        if not field_items:
            continue
            
        # Get the summary for this field
        field_summary = field_summaries.get(field_name)
        confidence_color = {
            "Very High": "green",
            "High": "lightgreen", 
            "Medium": "orange",
            "Low": "red"
        }.get(field_summary.confidence if field_summary else "Low", "gray")
        
        # Create expandable section with confidence indicator
        expander_title = f"{field_name} ({len(field_items)} items)"
        if field_summary:
            expander_title += f" - Confidence: {field_summary.confidence}"
        
        with st.expander(expander_title, expanded=False):
            # If we have an AI summary, display it first
            if field_summary:
                st.markdown("### Summary")
                st.write(field_summary.summary)
                
                if field_summary.key_items:
                    st.markdown("### Key Items")
                    for item in field_summary.key_items:
                        st.markdown(f"- {item}")
                
                st.divider()
            
            # Handle different field types with appropriate formatting
            if field_name == "Products / Services":
                # Display product services in a table format
                product_data = []
                for item in field_items:
                    product = item['data']
                    if isinstance(product, BaseModel):
                        product_dict = product.model_dump() if hasattr(product, 'model_dump') else product.dict()
                        product_dict['Source'] = item['source']
                        product_data.append(product_dict)
                
                if product_data:
                    st.dataframe(pd.DataFrame(product_data))
                    
            elif field_name == "Investors":
                # Display investors in a specialized format
                for item in field_items:
                    investor_data = item['data']
                    if hasattr(investor_data, 'investors'):
                        for investor in investor_data.investors:
                            st.write(f"**{investor.investor_name}**")
                            if investor.investor_type:
                                st.write(f"Type: {investor.investor_type}")
                            if investor.location:
                                st.write(f"Location: {investor.location}")
                            st.divider()
                    
            elif field_name == "Year Founded":
                # Simple display for year founded
                years = []
                for item in field_items:
                    if hasattr(item['data'], 'year_founded'):
                        years.append(item['data'].year_founded)
                
                if years:
                    from collections import Counter
                    year_counts = Counter(years)
                    for year, count in year_counts.most_common():
                        st.write(f"**{year}**: {count} sources")
                        
            else:
                # Generic handling for other field types
                for item in field_items:
                    data = item['data']
                    st.write(f"**Source**: [{item['source']}]({item['source']})")
                    
                    if isinstance(data, BaseModel):
                        # Display Pydantic model
                        model_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
                        for key, value in model_dict.items():
                            if value:  # Skip empty values
                                st.write(f"**{key}**: {value}")
                    elif isinstance(data, dict):
                        # Display dictionary
                        for key, value in data.items():
                            st.write(f"**{key}**: {value}")
                    else:
                        # Display other data types
                        st.write(data)
                    
                    st.divider()


# At the beginning of display_interactive_results function
async def display_interactive_results(selected_company, selection_results, skip_non_exact=False):
    """
    Displays search results with tabbed interface and data editor for selections.
    """
    if st.session_state.selected_company and st.session_state.search_results_all_companies.get(st.session_state.selected_company):
        company_features = st.session_state.search_results_all_companies[st.session_state.selected_company].get("company_features", None)
        cleaned_search_query_name = st.session_state.search_results_all_companies[st.session_state.selected_company].get("cleaned_search_query_name")
        results_with_metadata = st.session_state.search_results_all_companies[st.session_state.selected_company].get("results_with_metadata", [])
        fixed_results = fix_missing_search_queries(results_with_metadata)
        st.session_state.search_results_all_companies[st.session_state.selected_company]["results_with_metadata"] = fixed_results
        non_exact_match_results_metadata = st.session_state.search_results_all_companies[st.session_state.selected_company].get("non_exact_match_results_metadata", [])
        # Safely get negative_examples_collection or set to None
        negative_examples_collection = st.session_state.search_results_all_companies[st.session_state.selected_company].get("negative_examples_collection", None)
    else:
        company_features = None
        cleaned_search_query_name = "Unknown Company"
        results_with_metadata = []
        non_exact_match_results_metadata = []
    
    # Initialize selection containers
    selected_results_exact = []
    selected_company_data_list_exact = []
    selected_count_exact = 0
    
    selected_results_non_exact = []
    selected_company_data_list_non_exact = []
    selected_count_non_exact = 0
    
    interactive_results_container = st.container()
    with interactive_results_container:
        if not results_with_metadata:
            st.info("No search results to display interactively.")
            return
        
        # Create main tabs for better organization
        overview_tab, exact_match_tab, potential_match_tab, sources_tab = st.tabs([
            "Company Overview", 
            "Exact Matches", 
            "Potential Matches", 
            "Source Analysis"
        ])
        
        with overview_tab:
            # Display search queries summary and company overview
            display_search_queries_summary(cleaned_search_query_name, results_with_metadata)
            
            # Company Overview and Features
            st.header("Company Overview and Features")
            if company_features:
                st.write("This section provides an overview and key features of the company based on aggregated search results.")
                
                features_data = {
                    "Feature": [],
                    "Value": []
                }
                
                input_company_name_for_display = cleaned_search_query_name
                
                features_data["Feature"].append("Company Name Searched")
                features_data["Value"].append(input_company_name_for_display)
                
                feature_mapping = {
                    "Company Overview": company_features.company_overview_summary,
                    "Industry": company_features.industry_overview,
                    "Products/Services": company_features.product_service_overview,
                    "Technology Platform": company_features.technology_platform_overview,
                    "Mission/Vision Statement": company_features.mission_vision_statement,
                    "Target Audience/Customers": company_features.target_audience_customers,
                    "Geographic Focus": company_features.geographic_focus,
                    "Organization Type": company_features.organization_type
                }
                
                for feature, value in feature_mapping.items():
                    if value:
                        features_data["Feature"].append(feature)
                        features_data["Value"].append(", ".join(value) if isinstance(value, list) else value)
                
                if features_data["Feature"]:
                    st.table(pd.DataFrame(features_data))
                else:
                    st.info("No specific features extracted from the search results.")
            else:
                st.info("No company features extracted from the search results.")
        
        with exact_match_tab:
            st.header(f"Exact Match Results for: {cleaned_search_query_name.title()}")
            
            # Display exact matches using data editor
            exact_matches = [
                result_dict for result_dict in results_with_metadata
                if (result_dict.get("extracted_company_name") and cleaned_search_query_name and 
                    result_dict.get("extracted_company_name").lower() == cleaned_search_query_name.lower())
            ]
            
            if exact_matches:
                # Use ONLY the data editor for selection
                selected_results_exact, selected_count_exact = render_exact_match_selection(exact_matches, cleaned_search_query_name)
                
                # Extract company data from selected results
                selected_company_data_list_exact = [
                    result["company_data"] for result in selected_results_exact
                    if result.get('company_data')
                ]
            else:
                st.info("No exact matches found.")
        
        with potential_match_tab:
            # Call the enhanced function that uses data editor
            if not skip_non_exact:
                await display_interactive_non_exact_matches_v4(selected_company)
            
            # Get selection data from session state
            if 'user_selected_non_exact' in st.session_state and st.session_state.selected_company in st.session_state.user_selected_non_exact:
                selections = st.session_state.user_selected_non_exact[st.session_state.selected_company]
                
                # Collect selected non-exact matches
                selected_results_non_exact = []
                selected_company_data_list_non_exact = []
                
                for idx, selected in selections.items():
                    if selected and int(idx) < len(non_exact_match_results_metadata):
                        result = non_exact_match_results_metadata[int(idx)]
                        selected_results_non_exact.append(result)
                        if result.get('company_data'):
                            selected_company_data_list_non_exact.append(result['company_data'])
                
                selected_count_non_exact = len(selected_company_data_list_non_exact)
        
        with sources_tab:
            st.header("Source Analysis")
            
            # Organize sources by domain
            domains = {}
            all_metadata = []
            
            for result in results_with_metadata:
                metadata = result.get('search_result_metadata', {})
                if metadata:
                    domain = get_domain(metadata.get('url', ''))
                    if domain not in domains:
                        domains[domain] = 0
                    domains[domain] += 1
                    all_metadata.append({
                        "Domain": domain,
                        "Title": metadata.get('title', 'No title'),
                        "URL": metadata.get('url', 'No URL'),
                        "Query": metadata.get('search_query', 'Unknown query'),
                        "Extracted Company": result.get('extracted_company_name', 'Not extracted')
                    })
            
            # Display source distribution
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Source Distribution")
                domain_df = pd.DataFrame({"Domain": list(domains.keys()), "Count": list(domains.values())})
                domain_df = domain_df.sort_values("Count", ascending=False).reset_index(drop=True)
                st.dataframe(domain_df)
            
            with col2:
                st.subheader("Search Query Distribution")
                queries = {}
                for result in results_with_metadata:
                    query = result.get('search_result_metadata', {}).get('search_query', 'Unknown')
                    if query not in queries:
                        queries[query] = 0
                    queries[query] += 1
                
                query_df = pd.DataFrame({"Query": list(queries.keys()), "Results": list(queries.values())})
                query_df = query_df.sort_values("Results", ascending=False).reset_index(drop=True)
                st.dataframe(query_df)
            
            # Display all sources table
            st.subheader("All Sources")
            metadata_df = pd.DataFrame(all_metadata)
            st.dataframe(metadata_df, use_container_width=True)
        # Generate table button (outside tabs)
        button_container = st.container()
        with button_container:
            if "field_specific_data" not in st.session_state:
                st.session_state.field_specific_data = {}

            combined_count = len(selected_company_data_list_exact) + len(selected_company_data_list_non_exact)
            if st.button(f"Generate Aggregated Table from {combined_count} Selected Sources", key=f"generate_table_button_{cleaned_search_query_name}"):
                # Use the already collected selections instead of re-collecting them
                combined_company_data_list = selected_company_data_list_exact + selected_company_data_list_non_exact
                
                # Get metadata for sources
                combined_metadata_list = [
                    result.get('search_result_metadata', {}) for result in selected_results_exact
                ] + [
                    result.get('search_result_metadata', {}) for result in selected_results_non_exact
                ]
                
                combined_results_container = st.container()
                
                with combined_results_container:
                    # Debug information for troubleshooting
                    with st.expander("Selection Debug Information", expanded=False):
                        st.write(f"Total exact match results: {len(results_with_metadata)}")
                        st.write(f"Total non-exact match results: {len(non_exact_match_results_metadata)}")
                        st.write(f"Selected exact match results: {len(selected_results_exact)}")
                        st.write(f"Selected non-exact match results: {len(selected_results_non_exact)}")
                        st.write(f"Combined selected results: {len(selected_results_combined)}")
                        
                        # Show checkbox keys
                        checkbox_keys = [key for key in st.session_state.keys() if "checkbox" in key]
                        st.write("Sample checkbox keys in session state:", checkbox_keys[:5] if checkbox_keys else "None found")
                    
                    # Fallback if no sources were selected
                    if not combined_company_data_list:
                        st.warning("No sources were detected as selected. Using a fallback selection method.")
                        
                        # Try alternative approach - select first 3 exact matches as fallback
                        if results_with_metadata:
                            selected_results_combined = results_with_metadata[:min(3, len(results_with_metadata))]
                            st.info(f"Selected {len(selected_results_combined)} sources as fallback.")
                            
                            # Extract company data from fallback selection
                            combined_company_data_list = [
                                result["company_data"] for result in selected_results_combined if result.get("company_data")
                            ]
                            combined_metadata_list = [
                                result.get('search_result_metadata', {}) for result in selected_results_combined
                            ]
                    
                    # Continue with displaying aggregated table and extraction
                    if combined_company_data_list:
                        # Display aggregated table
                        await display_table_results(
                            combined_company_data_list, 
                            cleaned_search_query_name, 
                            combined_metadata_list
                        )
                        
                        # Create status container for extraction process
                        extraction_status_container = st.container()
                        
                        # Extract field-specific data
                        field_specific_data = await extract_all_fields_direct(
                            selected_results_combined, 
                            extraction_status_container
                        )
                        st.session_state.field_specific_data = field_specific_data

                        # Display field data with source attribution using the improved UI
                        await display_field_specific_data_with_source_attribution(field_specific_data, extraction_status_container)
                    else:
                        st.error("No sources available for extraction even after fallback selection. Please try selecting different sources.")

class MatchItem(BaseModel):
    """A structured match item compatible with Gemini."""
    name: str = Field(..., description="Name of the entity")
    percentage: int = Field(..., description="Match percentage")
    matching_attributes: List[str] = Field(default_factory=list, description="Matching attributes")
    mismatching_attributes: List[str] = Field(default_factory=list, description="Mismatching attributes")

class NegativeMatchItem(BaseModel):
    """A structured negative match item compatible with Gemini."""
    name: str = Field(..., description="Name of the entity")
    reason: str = Field(..., description="Reason for negative match")

class GeminiCompatibleSelectionOutput(BaseModel):
    """Structured output for entity matching compatible with Gemini."""
    include_in_table: bool = Field(..., description="Whether to include this result in the table")
    reason: str = Field(..., description="Reasoning for including or excluding this search result.")  # Added this field
    critical_matches: List[str] = Field(default_factory=list, description="Critical matching attributes")
    closest_matches: List[MatchItem] = Field(default_factory=list, description="Ranked list of similar entities")
    negative_matches: List[NegativeMatchItem] = Field(default_factory=list, description="Entities that are not matches")
    missing_information: List[str] = Field(default_factory=list, description="Information needed to improve matching")
    overall_match_percentage: Optional[int] = Field(None, description="Overall match percentage")
    
gemini_selection_agent = Agent(
    model=gemini_2o_model,
    result_type=GeminiCompatibleSelectionOutput,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""You are an expert AI analyst determining if a search result should be included in a company information table.

Analyze whether this search result is about the target company and provide a STRUCTURED assessment:

1. CRITICAL MATCHES: List exact matches for critical identifiers (exact name, website, founding details)

2. CLOSEST MATCHES: For partial matches, provide a ranked list with match percentage and specific attributes:
   - For each match include: name, percentage (e.g., 75), matching attributes list, and mismatching attributes list

3. NEGATIVE MATCHES: List entities that are definitely NOT the target company with reasons
   - For each negative match include: name and specific reason

4. ADDITIONAL INFORMATION NEEDED: List specific information items that would help improve matching

5. REASON: Provide a clear, concise explanation for your inclusion/exclusion decision in the 'reason' field.

Your output must include a final decision on whether to include this result (include_in_table: true/false).
"""
)

# Function to format selection reasoning in a structured way for display
def format_structured_selection_reason(selection_result):
    """
    Formats selection reasoning in a structured way.
    
    Args:
        selection_result: The selection result from the structured_selection_agent
        
    Returns:
        Formatted string with structured selection reasoning
    """
    if not selection_result or not hasattr(selection_result, 'data'):
        return "*No structured selection reasoning available*"
    
    data = selection_result.data
    
    # Build formatted output
    formatted_text = []
    
    # Add critical matches
    formatted_text.append("**CRITICAL MATCHES:**")
    if data.critical_matches:
        for match in data.critical_matches:
            formatted_text.append(f"- {match}")
    else:
        formatted_text.append("- None found")
    
    # Add closest matches
    formatted_text.append("\n**CLOSEST MATCHES:**")
    if data.closest_matches:
        for i, match in enumerate(data.closest_matches):
            if isinstance(match, dict) and 'name' in match and 'percentage' in match:
                formatted_text.append(f"{i+1}. {match['name']} ({match['percentage']}% match):")
                
                if 'matching' in match:
                    for attr in match['matching']:
                        formatted_text.append(f"   ✓ {attr}")
                        
                if 'mismatching' in match:
                    for attr in match['mismatching']:
                        formatted_text.append(f"   ✗ {attr}")
    else:
        formatted_text.append("- No close matches found")
    
    # Add negative matches
    formatted_text.append("\n**NEGATIVE MATCHES:**")
    if data.negative_matches:
        for match in data.negative_matches:
            if isinstance(match, dict) and 'name' in match and 'reason' in match:
                formatted_text.append(f"- {match['name']}: {match['reason']}")
            elif isinstance(match, str):
                formatted_text.append(f"- {match}")
    else:
        formatted_text.append("- No negative matches identified")
    
    # Add missing information
    formatted_text.append("\n**ADDITIONAL INFORMATION NEEDED:**")
    if data.missing_information:
        formatted_text.append("To improve matching, please provide:")
        for info in data.missing_information:
            formatted_text.append(f"- {info}")
    else:
        formatted_text.append("- No additional information required")
    
    # Add overall match percentage if available
    if data.overall_match_percentage is not None:
        formatted_text.append(f"\n**OVERALL MATCH: {data.overall_match_percentage}%**")
    
    # Add decision
    decision = "Include in results" if data.include_in_table else "Exclude from results"
    formatted_text.append(f"\n**DECISION: {decision}**")
    
    return "\n".join(formatted_text)

async def display_interactive_non_exact_matches_v4(selected_company):
    """
    Enhanced version of display_interactive_non_exact_matches using st.data_editor
    with more robust session state management for persistent selections.
    """
    st.header("Potentially Relevant Sources for Aggregation")
    st.write("Select non-exact matches that contain relevant information about the target company.")
    
    # Prepare the data for data_editor
    non_exact_data = []
    
    if selected_company not in st.session_state.search_results_all_companies:
        st.info(f"No results found for {selected_company}")
        return
    
    company_results = st.session_state.search_results_all_companies[selected_company]
    non_exact_match_results = company_results.get('non_exact_match_results_metadata', [])
    
    if not non_exact_match_results:
        st.info("No non-exact matches found for this company.")
        return
    
    st.header("Non-Exact Match Review")
    st.write("Review potential matches that don't have an exact name match but may contain relevant information.")
    
    # Get entity verification results
    entity_verification_results = st.session_state.get('entity_verification_results', {}).get(selected_company, {})
    
    # Get semantic validation results
    semantic_validations = st.session_state.get('all_companies_semantic_validation', {}).get(selected_company, {})
    
    # Get negative examples collection
    negative_examples = company_results.get('negative_examples_collection')
    has_negative_examples = (negative_examples and 
                            hasattr(negative_examples, 'negative_examples') and 
                            negative_examples.negative_examples)
    
    # Initialize selection state in session state if not present
    selection_key = f"non_exact_selections_{selected_company}"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}
    
    # Prepare data for the data editor
    non_exact_data = []
    
    for idx, result in enumerate(non_exact_match_results):
        # Skip None results
        if result is None:
            continue
            
        result_metadata = result.get('search_result_metadata', {})
        extracted_name = result.get('extracted_company_name', 'Unknown')
        
        # IMPORTANT FIX: Skip results with None extracted_name
        if extracted_name is None:
            continue
        
        # Extract publication date using the utility function
        publication_date = extract_publication_date(result_metadata)
        
        # Determine verification status
        verification_status = "Not Verified"
        verification_confidence = "N/A"
        relationship = "Unknown"
        matching_attrs = []
        differentiating_attrs = []
        match_reasoning = "No detailed reasoning available"
        
        # Check entity verification
        if entity_verification_results and isinstance(entity_verification_results, dict) and extracted_name in entity_verification_results:
            verification = entity_verification_results[extracted_name]
            verification_status = "Same Entity" if verification.are_same_entity else "Different Entity"
            verification_confidence = verification.confidence_level if hasattr(verification, 'confidence_level') else "N/A"
            relationship = verification.relationship_type if hasattr(verification, 'relationship_type') and verification.relationship_type else "N/A"
            
            # Directly access the matching and differentiating attributes
            if hasattr(verification, 'matching_attributes'):
                matching_attrs = verification.matching_attributes
            
            if hasattr(verification, 'differentiating_attributes'):
                differentiating_attrs = verification.differentiating_attributes
                
            # If the attributes don't exist as properties, try to access them from a model_dump method
            if not matching_attrs and hasattr(verification, 'model_dump'):
                dump_data = verification.model_dump()
                matching_attrs = dump_data.get('matching_attributes', [])
                differentiating_attrs = dump_data.get('differentiating_attributes', [])
            
            # Final fallback: try to access as dictionary keys
            if not matching_attrs and isinstance(verification, dict):
                matching_attrs = verification.get('matching_attributes', [])
                differentiating_attrs = verification.get('differentiating_attributes', [])
            
            if hasattr(verification, 'explanation'):
                match_reasoning = verification.explanation
        
        # Check semantic validation if no entity verification
        elif semantic_validations and isinstance(semantic_validations, dict) and idx in semantic_validations:
            validation = semantic_validations[idx]
            verification_status = "Valid Match" if validation.is_valid_match else "Invalid Match"
            verification_confidence = validation.confidence_level if hasattr(validation, 'confidence_level') else "N/A"
            if hasattr(validation, 'negative_example_matches') and validation.negative_example_matches:
                verification_status = "Negative Match"
            if hasattr(validation, 'is_related_entity') and validation.is_related_entity:
                relationship = validation.relationship_type if hasattr(validation, 'relationship_type') else "Related"
            
            # Try to get matching/differentiating attributes from semantic validation
            if hasattr(validation, 'recommended_fields'):
                matching_attrs = validation.recommended_fields
            if hasattr(validation, 'detected_critical_conflicts'):
                differentiating_attrs = validation.detected_critical_conflicts
            
            # Extract reasoning if available
            if hasattr(validation, 'match_reasoning'):
                match_reasoning = validation.match_reasoning
        
        # Check negative examples collection
        negative_example_match = None
        negative_reasons = []
        if has_negative_examples:
            for example in negative_examples.negative_examples:
                # IMPORTANT FIX: Add null checks before calling lower()
                if (example is not None and 
                    hasattr(example, 'entity_name') and 
                    example.entity_name is not None and 
                    extracted_name is not None and
                    example.entity_name.lower() == extracted_name.lower()):
                    
                    negative_example_match = example
                    if hasattr(example, 'key_differentiators'):
                        differentiating_attrs.extend(example.key_differentiators)
                        negative_reasons = example.key_differentiators
                    verification_status = "Negative Example"
                    verification_confidence = example.confidence_score if hasattr(example, 'confidence_score') else "High"
                    break
        
        # Format matching and differentiating attributes for display
        # Ensure they're lists before processing
        if not isinstance(matching_attrs, list):
            matching_attrs = []
        if not isinstance(differentiating_attrs, list):
            differentiating_attrs = []
            
        # Safely convert attributes to strings and join them
        safe_matching_attrs = []
        for attr in matching_attrs:
            if attr is not None:
                safe_matching_attrs.append(str(attr))
        
        safe_differentiating_attrs = []
        for attr in differentiating_attrs:
            if attr is not None:
                safe_differentiating_attrs.append(str(attr))
                
        matching_str = ", ".join(safe_matching_attrs)
        if len(matching_attrs) > 3:
            matching_str += f" (+{len(matching_attrs) - 3} more)"
            
        differentiating_str = ", ".join(safe_differentiating_attrs)
        if len(differentiating_attrs) > 3:
            differentiating_str += f" (+{len(differentiating_attrs) - 3} more)"
        
        # If attributes are still empty, add a placeholder message
        if not matching_str:
            matching_str = "No specific matching attributes identified"
        if not differentiating_str:
            differentiating_str = "No specific differentiating attributes identified"
        
        # Format negative example reasons
        negative_reasons_str = ", ".join([str(r) for r in negative_reasons if r is not None])
        
        # CRITICAL FIX: Get stored selection state with better default value behavior
        # Use the persistent session state selection instead of recomputing selection status
        include = st.session_state[selection_key].get(str(idx), False)
        
        # Only apply automatic selection logic if this item hasn't been explicitly selected/deselected
        if str(idx) not in st.session_state[selection_key]:
            # Determine default selection based on verification status
            if verification_status == "Same Entity" and verification_confidence in ["High", "Very High"]:
                include = True
                st.session_state[selection_key][str(idx)] = True
            elif verification_status == "Valid Match" and verification_confidence in ["High", "Very High"]:
                include = True
                st.session_state[selection_key][str(idx)] = True
        
        # Create status icon for visual clarity
        status_icon = ""
        if verification_status == "Same Entity" or verification_status == "Valid Match":
            status_icon = "✅ "
        elif verification_status == "Different Entity" or verification_status == "Invalid Match" or verification_status == "Negative Match" or verification_status == "Negative Example":
            status_icon = "❌ "
        else:
            status_icon = "❓ "
        
        # Add negative example specific information
        negative_example_info = ""
        if negative_example_match:
            negative_example_info = f"NEGATIVE EXAMPLE: {negative_reasons_str}"
        
        # Add row to data
        non_exact_data.append({
            "Select": include,
            "Title": result_metadata.get('title', 'No title'),
            "Entity": extracted_name,
            "Verification": f"{status_icon}{verification_status} ({verification_confidence})",
            "Relationship": relationship,
            "Similar Findings": matching_str,
            "Dissimilar Findings": differentiating_str,
            "Negative Example": negative_example_info,
            "Reasoning": match_reasoning[:150] + "..." if len(match_reasoning) > 150 else match_reasoning,
            "URL": result_metadata.get('url', 'No URL'),
            "Index": idx  # Store index for reference
        })
    
    # If no data was collected, show a message and return
    if not non_exact_data:
        st.info("No valid non-exact match data available to display.")
        return
    
    # Convert to DataFrame
    non_exact_df = pd.DataFrame(non_exact_data)
    
    # Configure data editor with enhanced configuration for better display
    column_config = {
        "Select": st.column_config.CheckboxColumn(
            "Select",
            help="Select this source for inclusion in aggregated data",
            default=False
        ),
        "Title": st.column_config.TextColumn("Title", width="large"),
        "Entity": st.column_config.TextColumn("Entity Name", width="medium"),
        "Verification": st.column_config.TextColumn("Verification", width="medium"),
        "Relationship": st.column_config.TextColumn("Relationship", width="small"),
        "Similar Findings": st.column_config.TextColumn("Similar Findings", width="large"),
        "Dissimilar Findings": st.column_config.TextColumn("Dissimilar Findings", width="large"),
        "Negative Example": st.column_config.TextColumn("Negative Example", width="large"),
        "Reasoning": st.column_config.TextColumn("Verification Reasoning", width="large"),
        "URL": st.column_config.LinkColumn("Source URL", width="medium"),
        "Index": None  # Hide the index column
    }
    
    # CRITICAL FIX: Create a stable key that doesn't change between reruns
    # This avoids the re-creation of the widget that causes checkbox resets
    editor_key = f"non_exact_editor_{selected_company}"
    
    # Display data editor with expanded view to show all information at once
    edited_df = st.data_editor(
        non_exact_df,
        column_config=column_config,
        disabled=["Title", "Entity", "Verification", "Relationship", "Similar Findings", 
                 "Dissimilar Findings", "Negative Example", "Reasoning", "URL"],
        hide_index=True,
        use_container_width=True,
        key=editor_key
    )
    
    # CRITICAL FIX: Store selections in session state with better persistence
    for _, row in edited_df.iterrows():
        idx = int(row["Index"])
        st.session_state[selection_key][str(idx)] = row["Select"]
    
    # Update user_selected_non_exact for backward compatibility
    if 'user_selected_non_exact' not in st.session_state:
        st.session_state.user_selected_non_exact = {}
    
    if selected_company not in st.session_state.user_selected_non_exact:
        st.session_state.user_selected_non_exact[selected_company] = {}
    
    # Synchronize the two state models
    for idx_str, selected in st.session_state[selection_key].items():
        st.session_state.user_selected_non_exact[selected_company][int(idx_str)] = selected
    
    # Add button to update aggregated data with selection count indicator
    selected_count = sum(1 for _, selected in st.session_state[selection_key].items() if selected)
    if st.button(f"Update Aggregated Data with {selected_count} Selected Sources", key=f"update_button_{selected_company}"):
        update_aggregated_data_with_selections(selected_company)
        st.success("Aggregated data updated with your selections!")
        
        # Show count of selected sources
        st.info(f"You have selected {selected_count} non-exact matches for inclusion in aggregated data.")
        
        # Option to view updated results
        if st.button("View Updated Results", key=f"view_results_{selected_company}"):
            st.rerun()
            
    # Display negative examples in an expander
    if has_negative_examples and negative_examples.negative_examples:
        with st.expander("Detailed Negative Example Analysis", expanded=False):
            st.subheader("Confirmed Negative Examples")
            st.write("These entities have been identified as definitely NOT being the target company:")
            
            for example in negative_examples.negative_examples:
                if example is None or not hasattr(example, 'entity_name'):
                    continue
                    
                st.markdown(f"### {example.entity_name}")
                
                if hasattr(example, 'confidence_score'):
                    st.markdown(f"**Confidence:** {example.confidence_score}")
                
                if hasattr(example, 'similarity_to_target'):
                    st.markdown(f"**Similarity to target:** {example.similarity_to_target}")
                
                if hasattr(example, 'key_differentiators') and example.key_differentiators:
                    st.markdown("**Key differentiators:**")
                    for diff in example.key_differentiators:
                        if diff is not None:
                            st.markdown(f"- {diff}")
                
                st.markdown("---")

def render_exact_match_selection(exact_matches, cleaned_search_query_name):
    """
    Renders the exact match selection interface using st.data_editor.
    Returns the selected results.
    """
    st.subheader("Select Sources for Aggregation")
    
    # Prepare data for data editor
    exact_match_data = []
    
    for idx, result in enumerate(exact_matches):
        search_metadata = result.get('search_result_metadata', {})
        search_query = search_metadata.get('search_query', 'N/A')
        url = search_metadata.get('url', 'N/A')
        title = search_metadata.get('title', 'N/A')
        
        # Extract publication date using the utility function
        publication_date = extract_publication_date(search_metadata)
        
        # Get selection state from session state with default=True for exact matches
        key = f"exact_match_checkbox_{cleaned_search_query_name}_{idx}"
        is_selected = st.session_state.get(key, True)
        
        # Add to data
        exact_match_data.append({
            "Select": is_selected,
            "Title": title,
            "URL Domain": get_domain(url),
            "Date": publication_date,
            "Search Query": search_query,
            "Index": idx
        })
    
    # Configure data editor
    column_config = {
        "Select": st.column_config.CheckboxColumn(
            "Select", 
            help="Select this source for inclusion in aggregated data",
            default=True
        ),
        "Title": st.column_config.TextColumn("Title"),
        "URL Domain": st.column_config.TextColumn("Source Domain"),
        "Date": st.column_config.TextColumn("Publication Date"),
        "Search Query": st.column_config.TextColumn("Search Query"),
        "Index": None  # Hide index column
    }
    
    # Display data editor and get edited values
    edited_df = st.data_editor(
        pd.DataFrame(exact_match_data),
        column_config=column_config,
        disabled=["Title", "URL Domain", "Date", "Search Query"],
        hide_index=True,
        key=f"exact_match_editor_{cleaned_search_query_name}"
    )
    
    # Update session state with selections
    for _, row in edited_df.iterrows():
        idx = int(row["Index"])
        key = f"exact_match_checkbox_{cleaned_search_query_name}_{idx}"
        st.session_state[key] = row["Select"]
    
    # Return selected results
    selected_indices = [int(row["Index"]) for _, row in edited_df.iterrows() if row["Select"]]
    selected_results = [exact_matches[idx] for idx in selected_indices]
    
    return selected_results, len(selected_indices)

def extract_publication_date(metadata):
    """
    Extract publication date from search result metadata with robust error handling.
    
    Args:
        metadata: Dict containing search result metadata
        
    Returns:
        String containing the publication date or 'Not available'
    """
    # Check if metadata is None or not a dictionary
    if metadata is None or not isinstance(metadata, dict):
        return "Not available"
    
    # First check direct date fields in metadata
    if 'published_date' in metadata and metadata['published_date']:
        return str(metadata['published_date'])
    elif 'date' in metadata and metadata['date']:
        return str(metadata['date'])
    
    # Try to extract date from content if available
    if 'content' in metadata and metadata['content'] is not None:
        content = metadata['content']
        # Ensure content is a string
        if not isinstance(content, str):
            try:
                content = str(content)
            except:
                return "Not available"
        
        # Look for date patterns in the first 500 characters
        import re
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # MM/DD/YYYY or DD/MM/YYYY
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',    # YYYY/MM/DD
            r'([A-Z][a-z]{2,8} \d{1,2},? \d{4})' # Month DD, YYYY
        ]
        
        # Safely get content substring
        try:
            content_sample = content[:min(500, len(content))]
            for pattern in date_patterns:
                matches = re.search(pattern, content_sample)
                if matches:
                    return matches.group(1)
        except Exception:
            # If any error occurs during regex, just continue
            pass
    
    return "Not available"

def display_company_verification_ui(company_data):
    """
    Displays UI for verifying company names before processing.
    
    Args:
        company_data: List of dictionaries with company information
        
    Returns:
        Tuple of (verified_data, verification_complete)
    """
    st.subheader("Company Name Verification")
    st.write("Please verify the following company names before processing.")
    
    if 'company_verification' not in st.session_state:
        st.session_state.company_verification = {}
    
    verified_data = []
    all_verified = True
    
    for idx, company in enumerate(company_data):
        company_name = company["name"]
        company_key = f"company_{idx}"
        
        # Initialize verification status if not already set
        if company_key not in st.session_state.company_verification:
            st.session_state.company_verification[company_key] = {
                "verified": False,
                "name": company_name,
                "alternative_names": [],
                "selected_name": company_name
            }
        
        with st.expander(f"Company: {company_name}", expanded=not st.session_state.company_verification[company_key]["verified"]):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Display company information
                st.write(f"**Original Name:** {company_name}")
                
                # Display alternative names if available
                alternatives = st.session_state.company_verification[company_key]["alternative_names"]
                if alternatives:
                    st.write("**Potential Alternative Names:**")
                    for alt in alternatives:
                        st.write(f"- {alt}")
                
                # Allow user to edit the name
                new_name = st.text_input(
                    "Verified Company Name", 
                    value=st.session_state.company_verification[company_key]["selected_name"],
                    key=f"name_input_{idx}"
                )
                
                # Update the selected name
                st.session_state.company_verification[company_key]["selected_name"] = new_name
            
            with col2:
                # Verification checkbox
                verified = st.checkbox(
                    "Verified", 
                    value=st.session_state.company_verification[company_key]["verified"],
                    key=f"verified_checkbox_{idx}"
                )
                
                # Update verification status
                st.session_state.company_verification[company_key]["verified"] = verified
                
                if not verified:
                    all_verified = False
                
                # Fetch alternative names button
                if st.button("Find Alternatives", key=f"find_alt_button_{idx}"):
                    with st.spinner("Searching for alternative names..."):
                        # This would typically make an API call to find alternatives
                        # For now, we'll simulate with a placeholder
                        st.session_state.company_verification[company_key]["alternative_names"] = [
                            f"{company_name} Inc.",
                            f"{company_name} LLC",
                            f"{company_name} Technologies"
                        ]
                        st.rerun()
        
        # Build verified data list
        verified_data.append({
            "name": st.session_state.company_verification[company_key]["selected_name"],
            "urls": company["urls"],
            "row_data": company["row_data"],
            "original_name": company_name
        })
    
    # Summary
    if all_verified:
        st.success("All companies verified! You can proceed to batch processing.")
    else:
        st.warning(f"{len([c for c in company_data if not st.session_state.company_verification.get(f'company_{company_data.index(c)}', {}).get('verified', False)])} companies still need verification.")
    
    return verified_data, all_verified

def display_individual_search_results():
    """
    Displays all individual search results for each company instead of aggregated data.
    Creates a flattened view with one row per search result.
    """
    if 'search_results_all_companies' not in st.session_state:
        st.info("No search results available.")
        return
    
    # Create header
    st.subheader("All Individual Search Results")
    st.write("This view shows each individual search result instead of aggregated data.")
    
    # Create a list to hold all individual results
    all_results = []
    
    # Process each company's search results
    for company_name, company_data in st.session_state.search_results_all_companies.items():
        results_with_metadata = company_data.get('results_with_metadata', [])
        cleaned_search_query_name = company_data.get('cleaned_search_query_name', '')
        
        # Add each individual result as a row
        for result in results_with_metadata:
            search_metadata = result.get('search_result_metadata', {})
            extracted_company_data = result.get('company_data', {})
            
            # Safely handle extracted_company_name to avoid None errors
            extracted_name = result.get('extracted_company_name')
            extracted_name_str = "" if extracted_name is None else str(extracted_name).lower()
            
            # Safely handle cleaned_search_query_name
            cleaned_name_str = "" if cleaned_search_query_name is None else str(cleaned_search_query_name).lower()
            
            # Create a row for this result
            result_row = {
                "Company Name": company_name,
                "Result Title": search_metadata.get('title', 'No title'),
                "Result URL": search_metadata.get('url', 'No URL'),
                "Search Query": search_metadata.get('search_query', 'No query'),
                "Source Type": search_metadata.get('source_type', 'Unknown'),
                "Is Exact Match": extracted_name_str == cleaned_name_str
            }
            
            # Add company data if available
            if extracted_company_data:
                # Add selected fields from company data to avoid overwhelming the display
                fields_to_include = [
                    "company_name", "product_type", "scientific_domain", 
                    "organization_type", "description_abstract", "total_funding"
                ]
                
                for field in fields_to_include:
                    if field in extracted_company_data:
                        field_value = extracted_company_data[field]
                        # Handle list values by joining with commas
                        if isinstance(field_value, list):
                            # Safely handle potentially None items in the list
                            value_str = ", ".join(str(x) for x in field_value if x is not None)
                            result_row[field.replace("_", " ").title()] = value_str
                        else:
                            # Handle potentially None value
                            result_row[field.replace("_", " ").title()] = "" if field_value is None else str(field_value)
            
            all_results.append(result_row)
    
    # Create DataFrame from all results
    if all_results:
        try:
            results_df = pd.DataFrame(all_results)
            
            # Add styling
            def highlight_exact_matches(row):
                if row.get('Is Exact Match', False):
                    return ['background-color: #d4f7d4' for _ in row.index]
                return ['background-color: #ffe6e6' for _ in row.index]
            
            # Display with styling
            styled_df = results_df.style.apply(highlight_exact_matches, axis=1)
            st.dataframe(styled_df)
            
            # Add download button
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="Download All Individual Results as CSV",
                data=csv_data,
                file_name="all_search_results.csv",
                mime="text/csv"
            )
            
            # Show stats
            st.info(f"Found {len(all_results)} individual search results across {len(st.session_state.search_results_all_companies)} companies.")
        except Exception as e:
            st.error(f"Error creating results table: {str(e)}")
            
            # Fallback to simplified display
            st.write("Showing simplified results due to an error:")
            for i, result in enumerate(all_results[:20]):  # Show first 20 for simplicity
                with st.expander(f"Result {i+1}: {result.get('Result Title', 'Untitled')}"):
                    st.json(result)
            
            if len(all_results) > 20:
                st.write(f"...and {len(all_results) - 20} more results")
    else:
        st.warning("No individual search results found.")

def display_previously_processed_results():
    """Displays previously processed results with both aggregated and individual views."""
    try:
        st.subheader("Previously Processed Results")
        
        # Create tabs for different views
        result_tab1, result_tab2 = st.tabs(["Aggregated View", "Individual Results"])
        
        with result_tab1:
            st.subheader("Aggregated Results (One Row Per Company)")
            if 'final_df' in st.session_state:
                safe_display_dataframe(st.session_state.final_df)
                add_download_button(st.session_state.final_df)
            else:
                st.warning("No aggregated results available.")
        
        with result_tab2:
            display_individual_search_results()
        
        # Display negative examples summary
        if 'all_companies_negative_examples' in st.session_state:
            st.subheader("Negative Examples Summary")
            
            neg_examples_by_company = {}
            for company, neg_examples in st.session_state.all_companies_negative_examples.items():
                if neg_examples and hasattr(neg_examples, 'negative_examples') and neg_examples.negative_examples:
                    neg_examples_by_company[company] = len(neg_examples.negative_examples)
            
            if neg_examples_by_company:
                example_df = pd.DataFrame(list(neg_examples_by_company.items()), 
                                    columns=["Company", "Number of Negative Examples"])
                st.dataframe(example_df)
            else:
                st.info("No negative examples were identified in the processing.")
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")

async def process_all_companies_field_extraction(uploaded_file=None):
    """
    Process field extraction and aggregation for all batch-processed companies in parallel.
    This function runs after the main batch processing is complete.
    
    Args:
        uploaded_file: Optional Excel file (used only if needed to recreate DataFrame)
        
    Returns:
        Final pandas DataFrame with all aggregated company data
    """
    if 'search_results_all_companies' not in st.session_state:
        st.error("No batch processing results found. Please process companies first.")
        return None
    
    with st.status("Processing field extraction for all companies...", expanded=True) as status:
        status_container = st.container()
        progress_bar = status_container.progress(0.0)
        status_text = status_container.empty()
        
        # Get all processed companies
        all_companies = list(st.session_state.search_results_all_companies.keys())
        total_companies = len(all_companies)
        
        if total_companies == 0:
            status.update(label="No companies found in batch processing results", state="error")
            return None
            
        status_text.write(f"Processing field extraction for {total_companies} companies...")
        
        # Create extraction tasks for all companies
        extraction_tasks = []
        extraction_companies = []
        
        for company_name in all_companies:
            extraction_companies.append(company_name)
            extraction_tasks.append(
                extract_company_fields(
                    company_name=company_name,
                    company_data=st.session_state.search_results_all_companies[company_name]
                )
            )
        
        # Execute all extraction tasks in batches for optimal performance
        batch_size = 5  # Process 5 companies at a time to avoid rate limits
        all_extraction_results = []
        
        for i in range(0, len(extraction_tasks), batch_size):
            batch_tasks = extraction_tasks[i:i+batch_size]
            batch_companies = extraction_companies[i:i+batch_size]
            
            status_text.write(f"Processing batch {i//batch_size + 1}/{math.ceil(len(extraction_tasks)/batch_size)}: {', '.join(batch_companies)}")
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_extraction_results.extend(batch_results)
            
            # Update progress
            progress = min(1.0, (i + len(batch_tasks)) / len(extraction_tasks))
            progress_bar.progress(progress)
        
        # Process extraction results and update company data
        successful_extractions = 0
        failed_extractions = 0
        
        for i, result in enumerate(all_extraction_results):
            company_name = extraction_companies[i]
            
            if isinstance(result, Exception):
                failed_extractions += 1
                status_text.error(f"Failed to extract fields for {company_name}: {str(result)}")
            else:
                successful_extractions += 1
                # Store field data in company results
                st.session_state.search_results_all_companies[company_name]['field_data_extraction'] = result
        
        # Compile final dataframe with all extracted data
        status_text.write("Compiling final aggregated table...")
        
        try:
            # Get Excel DataFrame if available
            if 'excel_df' in st.session_state:
                df = st.session_state.excel_df
            elif uploaded_file:
                df = pd.read_excel(uploaded_file)
                st.session_state.excel_df = df
            else:
                # Create DataFrame from company names if Excel not available
                df = pd.DataFrame({"Company Name": all_companies})
            
            # Compile final data
            final_df = compile_final_data_with_field_extraction(
                df, 
                st.session_state.search_results_all_companies
            )
            
            # Store in session state
            st.session_state.final_df = final_df
            
            status.update(
                label=f"Completed field extraction: {successful_extractions} succeeded, {failed_extractions} failed", 
                state="complete"
            )
            
            return final_df
            
        except Exception as e:
            logger.error(f"Error compiling final table: {e}")
            logger.error(traceback.format_exc())
            status.update(label=f"Error compiling final table: {str(e)}", state="error")
            return None
        
async def extract_company_fields(company_name, company_data):
    """
    Extract field-specific data for a single company, with source selection logic
    matching the "Generate Aggregated Table" button behavior.
    
    Args:
        company_name: Name of the company
        company_data: Company data from search results
        
    Returns:
        Dictionary mapping field names to extracted data
    """
    try:
        # Get necessary data
        cleaned_search_query_name = company_data.get('cleaned_search_query_name', company_name)
        results_with_metadata = company_data.get('results_with_metadata', [])
        non_exact_match_results_metadata = company_data.get('non_exact_match_results_metadata', [])
        selection_results = company_data.get('selection_results', [])
        
        # Get semantic validation results if available
        semantic_validations = st.session_state.get('all_companies_semantic_validation', {}).get(company_name, {})
        
        # Select sources using same logic as the button
        # 1. First check for exact matches
        selected_results_exact = []
        
        # Use exact match checkboxes if available
        exact_match_keys = [
            key for key in st.session_state.keys() 
            if key.startswith(f"exact_match_checkbox_{cleaned_search_query_name}_")
        ]
        
        if exact_match_keys:
            # Use explicitly selected exact matches
            selected_results_exact = [
                result_dict
                for index, result_dict in enumerate(results_with_metadata) 
                if st.session_state.get(f"exact_match_checkbox_{cleaned_search_query_name}_{index}")
            ]
        else:
            # Otherwise use all exact matches based on name comparison
            selected_results_exact = [
                result_dict
                for result_dict in results_with_metadata 
                if (result_dict.get("extracted_company_name") and cleaned_search_query_name and 
                    result_dict.get("extracted_company_name").lower() == cleaned_search_query_name.lower())
            ]
        
        # 2. Then get non-exact matches
        selected_results_non_exact = []
        
        # Process each non-exact match
        for index_non_exact, result_dict in enumerate(non_exact_match_results_metadata):
            include_result = False
            
            # Check for explicit UI selection
            if st.session_state.get(f"non_exact_result_checkbox_{cleaned_search_query_name}_{index_non_exact}"):
                include_result = True
            
            # Check semantic validation results
            elif index_non_exact in semantic_validations:
                validation = semantic_validations[index_non_exact]
                # Include if high confidence and no negative examples
                if validation.is_valid_match and validation.confidence_level in ["High", "Very High"] and not validation.negative_example_matches:
                    include_result = True
            
            # Check selection agent results
            elif index_non_exact < len(selection_results) and selection_results[index_non_exact].data and selection_results[index_non_exact].data.include_in_table:
                include_result = True
            
            if include_result:
                selected_results_non_exact.append(result_dict)
        
        # Combine selected results
        selected_results_combined = selected_results_exact + selected_results_non_exact
        
        # Fallback: if nothing selected but we have exact matches, use those
        if not selected_results_combined and results_with_metadata:
            exact_matches = [
                result_dict
                for result_dict in results_with_metadata 
                if (result_dict.get("extracted_company_name") and cleaned_search_query_name and 
                    result_dict.get("extracted_company_name").lower() == cleaned_search_query_name.lower())
            ]
            
            if exact_matches:
                selected_results_combined = exact_matches[:min(3, len(exact_matches))]
            else:
                # Last resort: use first few results
                selected_results_combined = results_with_metadata[:min(3, len(results_with_metadata))]
        
        # Create a status container for this company (won't be displayed in batch mode)
        dummy_container = DummyStatusContainer()
        
        # Call extract_all_fields_direct (existing function)
        field_specific_data = await extract_all_fields_direct(
            selected_results_combined,
            dummy_container
        )
        
        return field_specific_data
        
    except Exception as e:
        logger.error(f"Error extracting fields for {company_name}: {e}")
        logger.error(traceback.format_exc())
        raise e
    
class DummyStatusContainer:
    """A dummy container that mimics Streamlit containers for batch mode processing."""
    
    def __init__(self):
        """Initialize the dummy container."""
        pass
        
    def write(self, *args, **kwargs):
        """Mimics st.write() but does nothing."""
        pass
        
    def progress(self, value=0):
        """Mimics st.progress() but returns a dummy object."""
        return self
        
    def empty(self):
        """Mimics st.empty() but returns self."""
        return self
        
    def success(self, *args, **kwargs):
        """Mimics st.success() but does nothing."""
        pass
        
    def info(self, *args, **kwargs):
        """Mimics st.info() but does nothing."""
        pass
        
    def warning(self, *args, **kwargs):
        """Mimics st.warning() but does nothing."""
        pass
        
    def error(self, *args, **kwargs):
        """Mimics st.error() but does nothing."""
        pass
        
    def expander(self, *args, **kwargs):
        """Mimics st.expander() but returns a context manager."""
        class DummyExpander:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def write(self, *args, **kwargs):
                pass
            def json(self, *args, **kwargs):
                pass
            def table(self, *args, **kwargs):
                pass
            def markdown(self, *args, **kwargs):
                pass
        return DummyExpander()
        
    def container(self):
        """Mimics st.container() but returns self."""
        return self
        
    def table(self, *args, **kwargs):
        """Mimics st.table() but does nothing."""
        pass
        
    def json(self, *args, **kwargs):
        """Mimics st.json() but does nothing."""
        pass
    
    def markdown(self, *args, **kwargs):
        """Mimics st.markdown() but does nothing."""
        pass
    
    def code(self, *args, **kwargs):
        """Mimics st.code() but does nothing."""
        pass
    
    def metric(self, *args, **kwargs):
        """Mimics st.metric() but does nothing."""
        pass
    
    def columns(self, *args, **kwargs):
        """Mimics st.columns() but returns a list of dummy containers."""
        return [self for _ in range(args[0] if args else 1)]

def updated_tab2_implementation_v3():
    """
    Updated implementation for Tab2 that includes parallel field extraction and aggregation.
    Only minimal changes to the existing function.
    """
    st.header("Batch Process Companies from Excel")
    
    # Restore previous uploaded file if available
    uploaded_file = st.file_uploader(
        "Upload Excel file with company names in the first column", 
        type=["xlsx", "xls"],
        key="batch_file_uploader"
    )
    
    # Update session state when file is uploaded
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
    
    # Display previously processed companies status if available
    if 'processing_companies' in st.session_state and st.session_state.get('processing_companies'):
        display_processing_status(st.session_state.processing_companies)
    
    batch_results_container = st.container()
    
    # Create tabs for verification and processing
    tab2_1, tab2_2 = st.tabs(["Company Verification", "Batch Processing"])
    
    if uploaded_file is not None:
        with tab2_1:
            if 'company_data' not in st.session_state or st.button("Re-verify Companies"):
                # Read Excel into DataFrame for verification
                try:
                    df = pd.read_excel(uploaded_file)
                    
                    # Extract company names from the first column
                    company_column = df.columns[0]
                    company_data = []
                    url_column_names = ["Website", "URL", "Company URL", "Company Website"]
                    
                    for row_dict in df.to_dict('records'):
                        company_name = row_dict.get(company_column)
                        if company_name and not pd.isna(company_name):
                            # Get company URLs if available
                            company_urls = None
                            for col_name in url_column_names:
                                if col_name in df.columns and row_dict.get(col_name) and not pd.isna(row_dict.get(col_name)):
                                    company_urls = [str(row_dict.get(col_name))]
                                    break
                            
                            company_data.append({
                                "name": str(company_name),
                                "urls": company_urls,
                                "row_data": row_dict
                            })
                    
                    st.session_state.company_data = company_data
                    st.session_state.excel_df = df
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")
            
            # Display company verification UI
            if 'company_data' in st.session_state:
                verified_data, verification_complete = display_company_verification_ui(st.session_state.company_data)
                st.session_state.verified_company_data = verified_data
                
                if verification_complete:
                    st.success("Company verification complete! Proceed to the Batch Processing tab.")
                else:
                    st.warning("Some companies need review. Please verify all companies before processing.")
        
        with tab2_2:
            # Check if verification is complete
            if 'verified_company_data' not in st.session_state:
                st.warning("Please verify companies in the Company Verification tab first.")
            else:
                # Add two columns for processing buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    # Initial company processing button (unchanged)
                    if st.button("Process Company Information"):
                        try:
                            with batch_results_container:
                                # Execute original batch processing function
                                final_df = safe_async_run(process_uploaded_excel_with_enhanced_tracking_v3(uploaded_file))
                                
                                if final_df is not None:
                                    st.success("Company information processing completed!")
                                    st.session_state.initial_processing_complete = True
                                    
                                    # Display summary statistics
                                    total_companies = len(final_df)
                                    exact_matches = sum(1 for company in st.session_state.search_results_all_companies.values()
                                                      if company.get('grouped_results_dict'))
                                    
                                    st.write(f"Processed {total_companies} companies, found exact matches for {exact_matches} companies.")
                                    st.info("Now you can extract and aggregate field-specific data for all companies.")
                        except Exception as e:
                            st.error(f"Error processing Excel file: {e}")
                            logger.error(f"Error in batch processing: {traceback.format_exc()}")
                
                # NEW BUTTON FOR FIELD EXTRACTION
                with col2:
                    # Only show this button if initial processing is done
                    if st.session_state.get('initial_processing_complete', False):
                        if st.button("Extract & Aggregate All Fields"):
                            try:
                                with batch_results_container:
                                    # Execute new parallel field extraction and aggregation
                                    final_df = safe_async_run(process_all_companies_field_extraction(uploaded_file))
                                    
                                    if final_df is not None:
                                        st.success("Field extraction and aggregation completed for all companies!")
                                        
                                        # Display sample of results
                                        with st.expander("Preview of Results", expanded=True):
                                            st.dataframe(final_df.head(10))
                                            
                                            # Display stats
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Total Companies", len(final_df))
                                            with col2:
                                                st.metric("Fields Extracted", len(final_df.columns) - 2)  # Exclude Company Name and Data Source
                                            with col3:
                                                st.metric("Data Points", len(final_df) * (len(final_df.columns) - 2))
                                            
                                            # Add download button
                                            add_download_button(final_df)
                                        
                                        # View full results button
                                        if st.button("View Full Results"):
                                            st.session_state.active_tab = "Results"
                                            st.rerun()
                            except Exception as e:
                                st.error(f"Error in field extraction: {e}")
                                logger.error(f"Error in field extraction: {traceback.format_exc()}")
                    else:
                        st.info("Complete company information processing first, then extract and aggregate fields.")
                
                # Display reuse button if previously processed
                if 'final_df' in st.session_state:
                    st.divider()
                    st.write("Previously processed data is available:")
                    
                    if st.button("View Previously Processed Results"):
                        with batch_results_container:
                            display_previously_processed_results()

def compile_final_data_with_field_extraction(df, all_companies_search_results):
    """
    Compile consolidated table with proper formatting from batch processing results.
    
    Args:
        df: Original Excel DataFrame
        all_companies_search_results: Dictionary of search results by company
        
    Returns:
        DataFrame with all companies and their extracted data
    """
    table_rows = []
    
    # Process each company from the original Excel
    company_column = df.columns[0]
    
    for _, row in df.iterrows():
        company_name = row[company_column]
        if not company_name or pd.isna(company_name):
            continue
            
        company_name = str(company_name)
        
        # Start with basic row data from Excel
        row_data = {"Company Name": company_name}
        
        # Add original Excel data if appropriate
        for col, value in row.items():
            if col != company_column and not pd.isna(value):
                row_data[col] = value
        
        # Check if we have processed data for this company
        if company_name in all_companies_search_results:
            company_results = all_companies_search_results[company_name]
            
            # 1. Add basic company data from exact/non-exact matches
            company_data_sources = []
            exact_match = False
            
            # Process grouped results (exact matches)
            if 'grouped_results_dict' in company_results:
                for entity_name, results in company_results['grouped_results_dict'].items():
                    for result in results:
                        if result.get('company_data'):
                            company_data_sources.append(result['company_data'])
                            exact_match = True
            
            # If no exact matches, check non-exact
            if not company_data_sources and 'non_exact_match_results_metadata' in company_results:
                for result in company_results['non_exact_match_results_metadata']:
                    if result.get('company_data'):
                        company_data_sources.append(result['company_data'])
            
            # Aggregate company data if available
            if company_data_sources:
                # Create aggregated company data
                aggregated_company_data, _ = safe_async_run(
                    get_aggregated_company_data(company_data_sources, from_exact_match=exact_match)
                )
                
                # Add all fields from aggregated data
                model_data = aggregated_company_data.model_dump() if hasattr(aggregated_company_data, 'model_dump') else aggregated_company_data.dict()
                
                for field, value in model_data.items():
                    if value is not None:
                        # Format lists properly
                        if isinstance(value, list):
                            formatted_value = ", ".join(str(item) for item in value if item is not None)
                            if formatted_value:
                                row_data[field.replace("_", " ").title()] = formatted_value
                        # Format booleans properly
                        elif isinstance(value, bool):
                            if field == "female_co_founder":
                                row_data["Female Co-Founder"] = "Yes" if value else "No"
                            elif field.startswith("is_it_") and value:
                                # For funding stages and market cap booleans
                                field_label = field.replace("is_it_", "").replace("_", " ").title()
                                
                                if any(x in field for x in ["bootstrapped_low", "modest", "mega", "significant"]):
                                    # This is a market cap field
                                    if "Funding/Market Cap" not in row_data:
                                        row_data["Funding/Market Cap"] = field_label
                                    else:
                                        row_data["Funding/Market Cap"] += f", {field_label}"
                                else:
                                    # This is a funding stage field
                                    if "Funding Stage" not in row_data:
                                        row_data["Funding Stage"] = field_label
                                    else:
                                        row_data["Funding Stage"] += f", {field_label}"
                        else:
                            # Handle other types
                            row_data[field.replace("_", " ").title()] = value
            
            # 2. Add field-specific extracted data if available
            if 'field_data_extraction' in company_results:
                field_data = company_results['field_data_extraction']
                
                for field_name, field_items in field_data.items():
                    if not field_items:
                        continue
                        
                    # Process based on field type
                    if field_name == "Investors" and field_items:
                        investors = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'investors'):
                                for investor in data.investors:
                                    if hasattr(investor, 'investor_name') and investor.investor_name:
                                        investors.append(investor.investor_name)
                            elif isinstance(data, dict) and 'investors' in data:
                                for investor in data['investors']:
                                    if isinstance(investor, dict) and 'investor_name' in investor:
                                        investors.append(investor['investor_name'])
                        
                        if investors:
                            row_data["Investors"] = ", ".join(set(investors))
                    
                    elif field_name == "Products / Services" and field_items:
                        products = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'product_name') and data.product_name:
                                products.append(data.product_name)
                            elif isinstance(data, dict) and 'product_name' in data:
                                products.append(data['product_name'])
                        
                        if products:
                            row_data["Products"] = ", ".join(set(products))
                    
                    elif field_name == "Business Model(s)" and field_items:
                        models = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'business_models') and data.business_models:
                                models.extend(data.business_models)
                            elif isinstance(data, dict) and 'business_models' in data:
                                models.extend(data['business_models'])
                        
                        if models:
                            row_data["Business Model(s)"] = ", ".join(set(models))
                    
                    elif field_name == "HQ Location" and field_items:
                        locations = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'locations') and data.locations:
                                locations.extend(data.locations)
                            elif isinstance(data, dict) and 'locations' in data:
                                locations.extend(data['locations'])
                        
                        if locations:
                            row_data["HQ Location"] = ", ".join(set(locations))
                    
                    elif field_name == "Organization Type" and field_items:
                        org_types = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'organization_type') and data.organization_type:
                                org_types.append(data.organization_type)
                            elif isinstance(data, dict) and 'organization_type' in data:
                                org_types.append(data['organization_type'])
                        
                        if org_types:
                            row_data["Organization Type"] = ", ".join(set(org_types))
                    
                    elif field_name == "Year Founded" and field_items:
                        years = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'year_founded') and data.year_founded:
                                years.append(str(data.year_founded))
                            elif isinstance(data, dict) and 'year_founded' in data:
                                years.append(str(data['year_founded']))
                        
                        if years:
                            # Use most common year
                            from collections import Counter
                            year_counts = Counter(years)
                            most_common_year = year_counts.most_common(1)[0][0]
                            row_data["Year Founded"] = most_common_year
                    
                    elif field_name == "Female Co-Founder?" and field_items:
                        female_founders = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'female_co_founder'):
                                female_founders.append(data.female_co_founder)
                            elif isinstance(data, dict) and 'female_co_founder' in data:
                                female_founders.append(data['female_co_founder'])
                        
                        if female_founders:
                            # Use True if any sources say True
                            row_data["Female Co-Founder"] = "Yes" if any(female_founders) else "No"
                    
                    elif field_name == "Relevant Segments" and field_items:
                        segments = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'segments') and data.segments:
                                segments.extend(data.segments)
                            elif isinstance(data, dict) and 'segments' in data:
                                segments.extend(data['segments'])
                        
                        if segments:
                            row_data["Relevant Segments"] = ", ".join(set(segments))
                    
                    elif field_name == "Lab or Proprietary Data Generation?" and field_items:
                        data_gen_types = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'data_generation_types') and data.data_generation_types:
                                data_gen_types.extend([t for t in data.data_generation_types if t])
                            elif isinstance(data, dict) and 'data_generation_types' in data:
                                data_gen_types.extend([t for t in data['data_generation_types'] if t])
                        
                        if data_gen_types:
                            row_data["Lab/Proprietary Data"] = ", ".join(set(data_gen_types))
                    
                    elif field_name == "Drug Pipeline?" and field_items:
                        pipeline_stages = []
                        for item in field_items:
                            # IMPROVED TYPE CHECKING: Check if item is a dictionary before accessing keys
                            if not isinstance(item, dict):
                                continue
                                
                            data = item.get('data')
                            if data is None:
                                continue
                                
                            if hasattr(data, 'pipeline_stages') and data.pipeline_stages:
                                pipeline_stages.extend([s for s in data.pipeline_stages if s])
                            elif isinstance(data, dict) and 'pipeline_stages' in data:
                                pipeline_stages.extend([s for s in data['pipeline_stages'] if s])
                        
                        if pipeline_stages:
                            row_data["Drug Pipeline"] = ", ".join(set(pipeline_stages))
            
            # Add data source indicator
            row_data["Data Source"] = "Exact Match" if exact_match else "Non-Exact Match"
        else:
            # No processed data available
            row_data["Data Source"] = "Not Processed"
        
        # Add row to final table
        table_rows.append(row_data)
    
    # Create DataFrame
    final_df = pd.DataFrame(table_rows)
    
    # Ensure PyArrow compatibility for Streamlit display
    safe_df = sanitize_dataframe_values(final_df)
    
    # Reorder columns - put important ones first
    important_columns = [
        "Company Name", "Data Source", "Description Abstract", "Product Type", 
        "Scientific Domain", "Organization Type", "HQ Location", "Relevant Segments", 
        "Funding Stage", "Funding/Market Cap", "Year Founded", "Female Co-Founder"
    ]
    
    # Get columns that actually exist in the DataFrame
    existing_columns = [col for col in important_columns if col in safe_df.columns]
    other_columns = [col for col in safe_df.columns if col not in important_columns]
    
    # Reorder columns
    safe_df = safe_df[existing_columns + other_columns]
    
    return safe_df

def safe_display_dataframe(df, use_gradient=True):
    """
    Safely display a DataFrame in Streamlit with proper error handling for PyArrow conversion issues.
    
    Args:
        df: The pandas DataFrame to display
        use_gradient: Whether to apply a background gradient style
    """
    try:
        if use_gradient:
            # Check for columns with all NaN values and exclude them from styling
            numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
            cols_with_values = [col for col in numeric_cols if not df[col].isna().all()]
            
            if cols_with_values:
                # Only apply gradient to columns that have at least one non-NaN value
                styled_df = df.style.background_gradient(cmap='Blues', subset=cols_with_values)

                st.dataframe(styled_df)
            else:
                # If no suitable columns for gradient, just display without styling
                st.dataframe(df)
        else:
            st.dataframe(df)
    except Exception as e:
        # If we hit a PyArrow error or other display issue
        st.error(f"Error displaying formatted DataFrame: {str(e)}")
        
        try:
            # Try a simplified version without styling
            st.warning("Displaying simplified version without styling due to compatibility issues")
            
            # Create a simplified version by converting problematic columns
            simplified_df = df.copy()
            
            # Convert all object columns to strings
            for col in simplified_df.select_dtypes(include=['object']):
                simplified_df[col] = simplified_df[col].astype(str)
            
            # Try to display the simplified version
            st.dataframe(simplified_df)
            
        except Exception as e2:
            # If even the simplified version fails
            st.error(f"Could not display DataFrame: {str(e2)}")
            st.info("Attempting to display as a CSV...")
            
            # Last resort: convert to CSV and display as text
            csv_data = df.to_csv(index=False)
            st.text(csv_data)
            
            # Offer download option
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="company_research_results.csv",
                mime="text/csv"
            )



class AdaptiveRateLimiter:
    """
    Enhanced dynamic rate limiter that adjusts concurrency based on API response times and error rates.
    Helps prevent API rate limit errors by automatically throttling requests when needed.
    Includes improved error handling and monitoring.
    """
    def __init__(self, initial_limit=5, min_limit=1, max_limit=10):
        self.current_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.response_times = []
        self.error_count = 0
        self.consecutive_error_count = 0
        self.total_requests = 0
        self.success_requests = 0
        self.error_requests = 0
        self.semaphore = asyncio.Semaphore(initial_limit)
        self.lock = asyncio.Lock()
        self.last_adjustment_time = time.time()
        self.adjustment_cooldown = 5  # Seconds between adjustments
        
        # Initialize status tracking
        self.status_container = None
        self.status_text = "Initialized"
        
    def set_status_container(self, container):
        """Set a streamlit container to display status updates."""
        self.status_container = container
        
    def update_status(self, message):
        """Update status display if container is available."""
        self.status_text = message
        if self.status_container:
            self.status_container.info(message)
            
    async def acquire(self):
        """Acquire a slot from the semaphore with optional timeout."""
        self.total_requests += 1
        await self.semaphore.acquire()
        return time.time()  # Return acquisition time for timing
    
    def release(self, start_time=None, response_time=None, error=False):
        """
        Release the semaphore and adjust the rate limit based on metrics.
        
        Args:
            start_time: Optional time when acquire was called (for calculating response time)
            response_time: Optional time (in seconds) the API call took
            error: Whether an error occurred during the API call
        """
        # Calculate response time if not provided but start_time is
        if response_time is None and start_time is not None:
            response_time = time.time() - start_time
        
        if error:
            self.error_requests += 1
            self.error_count += 1
            self.consecutive_error_count += 1
            
            # Reduce limit immediately after consecutive errors
            if self.consecutive_error_count >= 3:
                asyncio.create_task(self._adjust_limit(decrease=True, reason="consecutive errors"))
                self.consecutive_error_count = 0
        else:
            self.success_requests += 1
            self.consecutive_error_count = 0
            
            if response_time:
                self.response_times.append(response_time)
                # Keep only the last 10 response times
                if len(self.response_times) > 10:
                    self.response_times.pop(0)
                    
                # Check if we should adjust based on response times
                self._check_response_times()
        
        # Release the semaphore
        self.semaphore.release()
    
    def _check_response_times(self):
        """Check response times and trigger adjustment if needed."""
        if len(self.response_times) < 3:
            return  # Need more data
            
        now = time.time()
        if now - self.last_adjustment_time < self.adjustment_cooldown:
            return  # Too soon to adjust again
            
        avg_time = sum(self.response_times) / len(self.response_times)
        
        if avg_time > 2.0:  # If average response time > 2 seconds
            asyncio.create_task(self._adjust_limit(decrease=True, reason=f"slow responses (avg {avg_time:.2f}s)"))
        elif avg_time < 0.5 and self.consecutive_error_count == 0:  # If fast and no errors
            # Only increase if success rate is good
            if self.total_requests > 10 and (self.success_requests / self.total_requests) > 0.9:
                asyncio.create_task(self._adjust_limit(decrease=False, reason=f"fast responses (avg {avg_time:.2f}s)"))
    
    async def _adjust_limit(self, decrease=True, reason="performance"):
        """
        Adjust the concurrency limit based on performance metrics.
        
        Args:
            decrease: Whether to decrease (True) or increase (False) the limit
            reason: Reason for the adjustment (for logging)
        """
        async with self.lock:
            self.last_adjustment_time = time.time()
            
            if decrease:
                new_limit = max(self.min_limit, self.current_limit - 1)
                adjustment = "decreased"
            else:
                new_limit = min(self.max_limit, self.current_limit + 1)
                adjustment = "increased"
            
            if new_limit != self.current_limit:
                # Create new semaphore with adjusted limit
                old_semaphore = self.semaphore
                old_limit = self.current_limit
                self.current_limit = new_limit
                self.semaphore = asyncio.Semaphore(new_limit)
                
                # Preserve acquired slots
                for _ in range(old_limit - old_semaphore._value):
                    await self.semaphore.acquire()
                
                self.update_status(f"Rate limit {adjustment} to {new_limit} concurrent requests due to {reason}")
                logger.info(f"API rate limit {adjustment} to {new_limit} concurrent requests due to {reason}")
    
    def get_stats(self):
        """Get current statistics about API usage."""
        success_rate = self.success_requests / self.total_requests if self.total_requests > 0 else 0
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "current_limit": self.current_limit,
            "available_slots": self.semaphore._value,
            "total_requests": self.total_requests,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "error_count": self.error_count,
            "status": self.status_text
        }
        
def update_aggregated_data_with_selections(company_name):
    """
    Updates the aggregated data for a company based on user selections of non-exact matches.
    
    Args:
        company_name: Name of the company to update data for
    """
    if 'user_selected_non_exact' not in st.session_state or company_name not in st.session_state.user_selected_non_exact:
        return
    
    # Get user selections
    selections = st.session_state.user_selected_non_exact[company_name]
    if not selections:
        return
    
    # Get company results
    company_results = st.session_state.search_results_all_companies.get(company_name, {})
    non_exact_match_results = company_results.get('non_exact_match_results_metadata', [])
    
    # Get the company index in the aggregated data list
    company_index = -1
    for i, input_name in enumerate(st.session_state.all_companies_original_inputs):
        if input_name == company_name:
            company_index = i
            break
    
    if company_index == -1:
        return
    
    # Collect data from selected non-exact matches
    selected_company_data = []
    
    # First, add any existing exact match data
    for entity_name, result_list in company_results.get('grouped_results_dict', {}).items():
        for result_dict in result_list:
            if result_dict.get('company_data'):
                selected_company_data.append(result_dict['company_data'])
    
    # Add selected non-exact match data
    for idx, selected in selections.items():
        if selected and int(idx) < len(non_exact_match_results):
            result = non_exact_match_results[int(idx)]
            if result.get('company_data'):
                selected_company_data.append(result['company_data'])
    
    # Update aggregated data
    if selected_company_data:
        # Get new aggregated data
        from_exact_match = len(company_results.get('grouped_results_dict', {})) > 0
        aggregated_company_data_object, _ = safe_async_run(get_aggregated_company_data(selected_company_data, from_exact_match=from_exact_match))
        
        # Update in session state
        st.session_state.all_companies_aggregated_data[company_index] = aggregated_company_data_object
        
        # Recompile final dataframe
        if 'excel_df' in st.session_state:
            df = st.session_state.excel_df
            final_df = compile_final_data_with_field_extraction(df, st.session_state.search_results_all_companies)
            st.session_state.final_df = final_df
            
#########################################
# Excel File Processing and UI Components
#########################################
def fix_batch_tracker_initialization(tracker):
    """
    Ensures that the BatchProcessTracker is properly initialized with all required keys.
    
    Args:
        tracker: BatchProcessTracker instance to check/fix
        
    Returns:
        Fixed BatchProcessTracker instance
    """
    # Check if company_progress is initialized
    if not hasattr(tracker, 'company_progress') or tracker.company_progress is None:
        tracker.company_progress = {}
    
    # For each company, ensure all required keys exist
    for company in tracker.companies:
        company_name = company["name"]
        
        # Initialize company_progress for this company if not exists
        if company_name not in tracker.company_progress:
            tracker.company_progress[company_name] = {
                "progress": 0.0,
                "status": "Pending",
                "last_update": None,
                "current_step": "Not Started",
                "log_messages": []
            }
        
        # Ensure container exists
        if "container" not in tracker.company_progress[company_name]:
            # Create a dummy container that won't throw errors
            class DummyContainer:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
                def markdown(self, *args, **kwargs):
                    pass
            
            tracker.company_progress[company_name]["container"] = DummyContainer()
            
        # Ensure bar exists
        if "bar" not in tracker.company_progress[company_name]:
            # Create a dummy progress bar that won't throw errors
            class DummyProgressBar:
                def progress(self, value):
                    pass
            
            tracker.company_progress[company_name]["bar"] = DummyProgressBar()
            
        # Ensure text exists
        if "text" not in tracker.company_progress[company_name]:
            # Create a dummy text container
            class DummyTextContainer:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
                def markdown(self, *args, **kwargs):
                    pass
                def write(self, *args, **kwargs):
                    pass
                def error(self, *args, **kwargs):
                    pass
                def warning(self, *args, **kwargs):
                    pass
                def info(self, *args, **kwargs):
                    pass
                def success(self, *args, **kwargs):
                    pass
            
            tracker.company_progress[company_name]["text"] = DummyTextContainer()
    
    return tracker

async def process_uploaded_excel_with_enhanced_tracking_v3(excel_file):
    """
    Further enhanced version of the batch processing function with optimized parallelization
    and improved error handling. Now includes detailed visual progress tracking.
    """
    # Read Excel into DataFrame
    try:
        df = pd.read_excel(excel_file)
        st.session_state.excel_df = df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None
    
    if df.empty:
        st.error("The uploaded Excel file is empty.")
        return None
    
    # Extract company names and URLs
    company_column = df.columns[0]
    company_data = []
    url_column_names = ["Website", "URL", "Company URL", "Company Website"]
    
    for row_dict in df.to_dict('records'):
        company_name = row_dict.get(company_column)
        if company_name and not pd.isna(company_name):
            # Get company URLs if available
            company_urls = None
            for col_name in url_column_names:
                if col_name in df.columns and row_dict.get(col_name) and not pd.isna(row_dict.get(col_name)):
                    company_urls = [str(row_dict.get(col_name))]
                    break
            
            company_data.append({
                "name": str(company_name),
                "urls": company_urls,
                "row_data": row_dict
            })
    
    if not company_data:
        st.error("No company names found in the Excel file.")
        return None
    
    # In your main processing function
    if 'active_tracker' in st.session_state:
        # Clear previous tracker UI
        for key in ['active_tracker', 'current_tracker']:
            if key in st.session_state:
                del st.session_state[key]    
    
    # Initialize the enhanced progress tracker
    tracker = BatchProcessTracker(company_data)
    tracker.initialize_ui()
    
    # Apply fix to ensure all required components are initialized
    tracker = fix_batch_tracker_initialization(tracker)
    
    st.session_state.active_tracker = tracker

    # OPTIMIZATION: Use a smaller initial limit for API calls and create a shared semaphore
    api_rate_limiter = AdaptiveRateLimiter(initial_limit=5)  # Reduced from 8

    
    # Add a global semaphore for file descriptor limitation
    # This is crucial to prevent "too many file descriptors" error
    global_semaphore = asyncio.Semaphore(25)  # Limit total concurrent operations
    
    # Initialize result containers
    all_companies_search_results = {}
    all_companies_aggregated_data = []
    all_companies_from_exact_match = []
    all_companies_original_inputs = []
    all_companies_semantic_validation = {}
    all_companies_negative_examples = {}
    
    # OPTIMIZATION: Process companies in smaller batches
    # Smaller batch size means fewer file descriptors open at once
    max_concurrent_companies = min(5, len(company_data))  # Reduced from 10
    
    # Group companies into batches
    company_batches = [company_data[i:i+max_concurrent_companies] 
                      for i in range(0, len(company_data), max_concurrent_companies)]
    
    batch_number = 1
    total_batches = len(company_batches)
    
    # Process each batch
    for batch in company_batches:
        tracker.overall_status.info(f"Processing batch {batch_number}/{total_batches}: {len(batch)} companies")
        
        # Create tasks for each company in the batch
        company_tasks = []
        
        for company in batch:
            company_name = company["name"]
            company_urls = company["urls"]
            
            # Set up processing steps as a single async task per company
            async def process_company(company_info, tracker, api_rate_limiter, global_semaphore):
                """Process a single company with robust error handling."""
                company_name = company_info["name"]
                company_urls = company_info["urls"]
                company_idx = tracker.companies.index(company_info) if company_info in tracker.companies else -1
                
                try:
                    # Start company tracking
                    c_name = tracker.start_company(company_idx)
                    
                    # Step 1: Search with robust error handling
                    tracker.update_step("Search", c_name)
                                        
                    # Execute the search with proper error handling and retries
                    max_retries = 3
                    retry_count = 0
                    search_results = None
                    
                    while retry_count < max_retries and search_results is None:
                        try:
                            await api_rate_limiter.acquire()
                            async with global_semaphore:
                                async with asyncio.timeout(120):  # 2-minute timeout for search
                                    # Using the enhanced search function with improved error handling
                                    # Pass cache parameters to search
                                    search_results = await safe_enhanced_search_company_summary_v4(
                                        company_name=company_name,
                                        company_urls=company_urls,
                                        use_test_data=False,
                                        NUMBER_OF_SEARCH_RESULTS=5,
                                        original_input=company_name,
                                        tracker=tracker,
                                        use_cache=st.session_state.use_cache,
                                        force_refresh=st.session_state.force_refresh
                                    )
                            api_rate_limiter.release()
                        except (httpx.ConnectTimeout, asyncio.TimeoutError) as e:
                            retry_count += 1
                            api_rate_limiter.release(error=True)
                            wait_time = retry_count * 2  # Exponential backoff
                            logger.error(f"Timeout in search for {company_name} (attempt {retry_count}/{max_retries}): {e}")
                            
                            # Create a failure notification in the progress display
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            with tracker.company_progress[company_name]["text"]:
                                st.warning(f"**{timestamp}**: Search attempt {retry_count} failed: Connection timeout. Retrying in {wait_time}s...")
                            
                            if retry_count < max_retries:
                                await asyncio.sleep(wait_time)
                            else:
                                # After max retries, return failure
                                tracker.complete_company(company_name, success=False)
                                return {
                                    "company_name": company_name,
                                    "success": False,
                                    "error": f"Search failed after {max_retries} attempts: {str(e)}"
                                }
                        except TypeError as e:
                            # Handle the specific TypeError we're seeing in the logs
                            if "'NoneType' object is not subscriptable" in str(e):
                                logger.error(f"TypeError in search for {company_name}: {e}")
                                api_rate_limiter.release(error=True)
                                # Don't retry this specific error as it's likely a data issue, not a connection issue
                                tracker.complete_company(company_name, success=False)
                                return {
                                    "company_name": company_name,
                                    "success": False,
                                    "error": f"Search failed with TypeError: {str(e)}"
                                }
                            else:
                                # For other TypeErrors, retry
                                retry_count += 1
                                api_rate_limiter.release(error=True)
                                wait_time = retry_count * 2
                                logger.error(f"TypeError in search for {company_name} (attempt {retry_count}/{max_retries}): {e}")
                                
                                if retry_count < max_retries:
                                    await asyncio.sleep(wait_time)
                                else:
                                    tracker.complete_company(company_name, success=False)
                                    return {
                                        "company_name": company_name,
                                        "success": False,
                                        "error": f"Search failed after {max_retries} attempts: {str(e)}"
                                    }
                        except Exception as e:
                            api_rate_limiter.release(error=True)
                            logger.error(f"Error in search for {company_name}: {e}")
                            logger.error(traceback.format_exc())
                            
                            # Create a failure notification
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            with tracker.company_progress[company_name]["text"]:
                                st.error(f"**{timestamp}**: Search failed: {str(e)}")
                            
                            tracker.complete_company(company_name, success=False)
                            return {
                                "company_name": company_name,
                                "success": False,
                                "error": str(e)
                            }

                    
                    # Check if search was successful
                    if not search_results:
                        tracker.complete_company(company_name, success=False)
                        return {
                            "company_name": company_name,
                            "success": False,
                            "error": "Search returned no results"
                        }
                    
                    # Unpack results including the new negative examples collection
                    (grouped_results_dict, cleaned_search_query_name, results_with_metadata, 
                     selection_results, non_exact_match_results_metadata, company_features,
                     negative_examples_collection) = search_results
                    
                    # Step 2: Extract & Process Data
                    tracker.update_step("Extract", c_name)
                    
                    # Update tracker with result count
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    with tracker.company_progress[company_name]["text"]:
                        st.markdown(f"**{timestamp}**: Found {len(results_with_metadata)} total results, {len(grouped_results_dict)} exact matches")
                    
                    company_results = {
                        "grouped_results_dict": grouped_results_dict,
                        "cleaned_search_query_name": cleaned_search_query_name,
                        "results_with_metadata": results_with_metadata,
                        "selection_results": selection_results,
                        "non_exact_match_results_metadata": non_exact_match_results_metadata,
                        "company_features": company_features,
                        "company_url": company_urls,
                        "negative_examples_collection": negative_examples_collection
                    }
                    
                    # OPTIMIZATION: Sequential processing with timeouts for verification and field search
                    
                    # Step 3: Verification with proper error handling
                    tracker.update_step("Verify", c_name)
                    try:
                        async with global_semaphore:
                            async with asyncio.timeout(90):  # 90-second timeout for verification
                                semantic_validation_results = await perform_semantic_verification(
                                    company_name, 
                                    cleaned_search_query_name, 
                                    results_with_metadata, 
                                    non_exact_match_results_metadata,
                                    company_features,
                                    negative_examples_collection,
                                    tracker=tracker,
                                    row_data=row_data  # Pass row data for verification
                                )
                        
                        # Special handling for no exact matches
                        if semantic_validation_results.get("_no_exact_matches", False):
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            with tracker.company_progress[company_name]["text"]:
                                st.warning(f"**{timestamp}**: No exact matches found. User selection required for entity verification.")
                        else:
                            # Update progress with verification results
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            with tracker.company_progress[company_name]["text"]:
                                st.markdown(f"**{timestamp}**: Verified {len(semantic_validation_results)} non-exact matches")
                    except Exception as e:
                        logger.error(f"Error in verification for {company_name}: {e}")
                        # Use empty results if verification fails
                        semantic_validation_results = {}
                        
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        with tracker.company_progress[company_name]["text"]:
                            st.warning(f"**{timestamp}**: Verification step encountered an error: {str(e)}")
                    
                    # Step 4: Field Search with proper error handling
                    tracker.update_step("Field Search", c_name)
                    try:
                        async with global_semaphore:
                            async with asyncio.timeout(90):  # 90-second timeout for field search
                                field_data_results = await perform_field_search(
                                    company_name,
                                    company_data[company_idx]["row_data"],
                                    cleaned_search_query_name,
                                    tracker=tracker
                                )
                        
                        # Update progress with field count
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        field_count = len(field_data_results) if field_data_results else 0
                        with tracker.company_progress[company_name]["text"]:
                            st.markdown(f"**{timestamp}**: Extracted data for {field_count} fields")
                    except Exception as e:
                        logger.error(f"Error in field search for {company_name}: {e}")
                        # Use empty results if field search fails
                        field_data_results = {}
                        
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        with tracker.company_progress[company_name]["text"]:
                            st.warning(f"**{timestamp}**: Field search encountered an error: {str(e)}")
                    
                    # Add results to company data
                    company_results["semantic_validation"] = semantic_validation_results
                    company_results["field_data_extraction"] = field_data_results
                    
                    # Step 5: Compile Data
                    tracker.update_step("Compile", c_name)
                    
                    # Determine aggregated data and match status
                    company_data_for_table_exact = []
                    for entity_name, result_list in grouped_results_dict.items():
                        for result_dict in result_list:
                            if result_dict.get('company_data'):
                                company_data_for_table_exact.append(result_dict['company_data'])
                    
                    if company_data_for_table_exact:
                        async with global_semaphore:
                            async with asyncio.timeout(30):  # 30-second timeout for aggregation
                                aggregated_company_data_object, from_exact_match = await get_aggregated_company_data(company_data_for_table_exact)
                        exact_match = True
                        
                        # Update progress with exact match success
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        with tracker.company_progress[company_name]["text"]:
                            st.markdown(f"**{timestamp}**: Successfully compiled data from exact matches")
                    else:
                        company_data_for_table_non_exact = []
                        
                        # Use semantically validated results
                        for i, result_dict in enumerate(non_exact_match_results_metadata):
                            include_result = False
                            
                            # Check if semantically validated
                            if i in semantic_validation_results and semantic_validation_results[i].is_valid_match:
                                # Check confidence level and negative example matches
                                if (semantic_validation_results[i].confidence_level in ["High", "Very High"] and 
                                        not semantic_validation_results[i].negative_example_matches):
                                    include_result = True
                                elif semantic_validation_results[i].is_related_entity:
                                    # Add with warning for related entities
                                    include_result = True
                            
                            # Also check agent selections
                            if i < len(selection_results) and selection_results[i].data and selection_results[i].data.include_in_table:
                                include_result = True
                            
                            if include_result and result_dict.get('company_data'):
                                company_data_for_table_non_exact.append(result_dict['company_data'])
                        
                        if company_data_for_table_non_exact:
                            async with global_semaphore:
                                async with asyncio.timeout(30):  # 30-second timeout
                                    aggregated_company_data_object, from_exact_match = await get_aggregated_company_data(
                                        company_data_for_table_non_exact, 
                                        from_exact_match=False
                                    )
                            exact_match = False
                            
                            # Update progress with non-exact match info
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            with tracker.company_progress[company_name]["text"]:
                                st.markdown(f"**{timestamp}**: Compiled data from non-exact matches ({len(company_data_for_table_non_exact)} sources)")
                        else:
                            # If no data found, create empty record
                            aggregated_company_data_object = CompanyDataOutput(company_name=company_name)
                            exact_match = False
                            
                            # Update progress with no-data warning
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            with tracker.company_progress[company_name]["text"]:
                                st.warning(f"**{timestamp}**: No usable data sources found")
                    
                    # Mark company as complete
                    tracker.complete_company(c_name, success=True)
                    
                    return {
                        "company_name": company_name,
                        "company_results": company_results,
                        "aggregated_data": aggregated_company_data_object,
                        "from_exact_match": exact_match,
                        "original_input": company_name,
                        "semantic_validation": semantic_validation_results,
                        "negative_examples": negative_examples_collection,
                        "success": True
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing company {company_name}: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Create a detailed error report in the tracker
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    try:
                        with tracker.company_progress[company_name]["text"]:
                            st.error(f"**{timestamp}**: Processing failed: {str(e)}")
                    except Exception:
                        # Fallback if there's an issue with the tracker
                        logger.error(f"Could not update tracker UI for company {company_name}")
                    
                    try:
                        tracker.complete_company(company_name, success=False)
                    except Exception as tracking_error:
                        logger.error(f"Error updating tracker for company {company_name}: {tracking_error}")
                    
                    return {
                        "company_name": company_name,
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }

            
            # Add task to batch
            company_tasks.append(process_company(company, tracker, api_rate_limiter, global_semaphore))

        
        # Process all companies in this batch concurrently
        batch_results = await asyncio.gather(*company_tasks)
        
        # Process results from this batch
        for result in batch_results:
            company_name = result["company_name"]
            
            if result["success"]:
                # Add successful results to containers
                all_companies_search_results[company_name] = result["company_results"]
                all_companies_aggregated_data.append(result["aggregated_data"])
                all_companies_from_exact_match.append(result["from_exact_match"])
                all_companies_original_inputs.append(result["original_input"])
                all_companies_semantic_validation[company_name] = result["semantic_validation"]
                all_companies_negative_examples[company_name] = result["negative_examples"]
            else:
                # Handle failed companies - add empty entries for consistency
                logger.error(f"Company {company_name} processing failed: {result.get('error')}")
                all_companies_aggregated_data.append(CompanyDataOutput(company_name=company_name))
                all_companies_from_exact_match.append(False)
                all_companies_original_inputs.append(company_name)
        
        # OPTIMIZATION: Clear unnecessary resources at the end of each batch
        import gc
        gc.collect()  # Force garbage collection
        batch_number += 1
    
    # Mark processing as complete
    tracker.complete_processing()
    
    # Store in session state for later use
    st.session_state.search_results_all_companies = all_companies_search_results
    st.session_state.all_companies_aggregated_data = all_companies_aggregated_data
    st.session_state.all_companies_from_exact_match = all_companies_from_exact_match
    st.session_state.all_companies_original_inputs = all_companies_original_inputs
    st.session_state.all_companies_semantic_validation = all_companies_semantic_validation
    st.session_state.all_companies_negative_examples = all_companies_negative_examples
    
    # Compile final data
    final_df = compile_final_data_with_field_extraction(df, all_companies_search_results)
    st.session_state.final_df = final_df
    
    # Return the final dataframe
    return final_df

# Helper functions for the enhanced batch processing

async def perform_semantic_verification(company_name, cleaned_search_query_name, results_with_metadata, 
                                       non_exact_match_results_metadata, company_features, 
                                       negative_examples_collection, tracker=None):
    """
    Optimized version of semantic verification with improved resource management and timeout handling.
    Updated to work with flat entity attributes and to handle the case of no exact matches.
    """
    try:
        # Skip further processing if there are no non-exact matches
        if not non_exact_match_results_metadata:
            return {}
            
        # Check for exact matches first 
        exact_matches = [
            result for result in results_with_metadata 
            if (result.get('extracted_company_name', '') and cleaned_search_query_name and 
                result.get('extracted_company_name', '').lower() == cleaned_search_query_name.lower() and
                result.get('search_result_metadata', {}).get('content'))
        ]
        
        # CRITICAL FIX: If no exact matches, return special status and skip verification
        if not exact_matches:
            # Track this situation if tracker provided
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Semantic Verification",
                    status="Skipped",
                    details="Skipped verification due to no exact matches available. User selection required."
                )
                
            # Return empty verification results with a special flag
            return {"_no_exact_matches": True}
            
        # Extract attributes for the target company with timeout, now including row data
        try:
            content_text = " ".join([
                result.get('search_result_metadata', {}).get('content', "") 
                for result in exact_matches
            ]) or ""
            
            # Apply timeout to entity attribute extraction
            async def extract_with_timeout():
                return await extract_entity_attributes(
                    content_text=content_text,
                    entity_name=cleaned_search_query_name or company_name,
                    row_data=row_data  # Pass row data to attribute extraction
                )
            
            target_company_attributes = await asyncio.wait_for(extract_with_timeout(), timeout=30)
            
            # Track attribute extraction if tracker provided
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Extract Target Attributes",
                    status="Completed",
                    details=f"Extracted attributes for target company: {cleaned_search_query_name or company_name}"
                )
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout extracting target company attributes for {company_name}")
            # Create minimal attributes as fallback
            target_company_attributes = FlatEntityAttributes(
                entity_name=cleaned_search_query_name or company_name,
                similar_names=[],
                founder_names=[],
                founder_mentions=[],
                founding_year=None,
                acquisition_year=None,
                rebranding_year=None,
                latest_known_activity=None,
                headquarters=None,
                additional_locations=[],
                total_funding_amount=None,
                latest_funding_round=None,
                investors=[],
                main_products=[],
                industry_focus=[],
                technologies=[],
                parent_company=None,
                subsidiaries=[],
                previous_names=[],
                related_entities=[]
            )
            
            # Track timeout if tracker provided
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Extract Target Attributes",
                    status="Timeout",
                    details=f"Timeout extracting attributes for target company: {cleaned_search_query_name or company_name}"
                )
                
        except Exception as e:
            logger.error(f"Error extracting target company attributes for {company_name}: {e}")
            # Create minimal attributes as fallback
            target_company_attributes = FlatEntityAttributes(
                entity_name=cleaned_search_query_name or company_name,
                similar_names=[],
                founder_names=[],
                founder_mentions=[],
                founding_year=None,
                acquisition_year=None,
                rebranding_year=None,
                latest_known_activity=None,
                headquarters=None,
                additional_locations=[],
                total_funding_amount=None,
                latest_funding_round=None,
                investors=[],
                main_products=[],
                industry_focus=[],
                technologies=[],
                parent_company=None,
                subsidiaries=[],
                previous_names=[],
                related_entities=[]
            )
            
            # Track error if tracker provided
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Extract Target Attributes",
                    status="Error",
                    details=f"Error extracting attributes for target company: {str(e)}"
                )
        
        # OPTIMIZATION: Limit concurrent verifications to avoid resource exhaustion
        # Use a semaphore to limit concurrent tasks
        sem = asyncio.Semaphore(5)  # Process at most 5 verifications at once
        
        # Perform enhanced semantic matching validation for non-exact matches
        company_semantic_validation = {}
        
        if company_features and non_exact_match_results_metadata:
            verification_tasks = []
            
            # Process in smaller batches
            max_verifications = min(10, len(non_exact_match_results_metadata))
            non_exact_batch = non_exact_match_results_metadata[:max_verifications]
            
            # Track verification process if tracker provided
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Semantic Verification",
                    status="Started",
                    details=f"Verifying {len(non_exact_batch)} non-exact matches"
                )
            
            for idx, result in enumerate(non_exact_batch):
                search_result_metadata = result.get('search_result_metadata', {})
                
                # Create task to extract attributes and perform differentiation/validation
                async def verify_non_exact_match(idx, result):
                    try:
                        async with sem:  # Use semaphore to limit concurrency
                            # Extract attributes with timeout
                            try:
                                # Apply timeout to entity attribute extraction
                                async def extract_comparison_with_timeout():
                                    return await extract_entity_attributes(
                                        content_text=search_result_metadata.get('content', ''),
                                        entity_name=result.get('extracted_company_name', 'Unknown')
                                    )
                                
                                search_result_attributes = await asyncio.wait_for(extract_comparison_with_timeout(), timeout=25)
                                
                                # Track attribute extraction if tracker provided
                                if tracker and company_name:
                                    extracted_name = result.get('extracted_company_name', 'Unknown')
                                    tracker.add_data_entry(
                                        company_name=company_name,
                                        operation="Extract Comparison Attributes",
                                        status="Completed",
                                        details=f"Extracted attributes for comparison entity: {extracted_name}",
                                        url=search_result_metadata.get('url', '')
                                    )
                                    
                            except (asyncio.TimeoutError, Exception) as e:
                                logger.warning(f"Error extracting comparison attributes: {e}")
                                # Create minimal attributes as fallback
                                search_result_attributes = FlatEntityAttributes(
                                    entity_name=result.get('extracted_company_name', 'Unknown'),
                                    similar_names=[],
                                    founder_names=[],
                                    founder_mentions=[],
                                    founding_year=None,
                                    acquisition_year=None,
                                    rebranding_year=None,
                                    latest_known_activity=None,
                                    headquarters=None,
                                    additional_locations=[],
                                    total_funding_amount=None,
                                    latest_funding_round=None,
                                    investors=[],
                                    main_products=[],
                                    industry_focus=[],
                                    technologies=[],
                                    parent_company=None,
                                    subsidiaries=[],
                                    previous_names=[],
                                    related_entities=[]
                                )
                                
                                # Track error if tracker provided
                                if tracker and company_name:
                                    extracted_name = result.get('extracted_company_name', 'Unknown')
                                    tracker.add_data_entry(
                                        company_name=company_name,
                                        operation="Extract Comparison Attributes",
                                        status="Error",
                                        details=f"Error extracting attributes for {extracted_name}: {str(e)}",
                                        url=search_result_metadata.get('url', '')
                                    )
                            
                            # Perform enhanced differentiation with timeout
                            try:
                                # Apply timeout to differentiation
                                async def differentiate_with_timeout():
                                    return await enhanced_differentiate_entities_v4(
                                        reference_attributes=target_company_attributes,
                                        comparison_attributes=search_result_attributes,
                                        content_text=search_result_metadata.get('content', '')
                                    )
                                
                                differentiation_result = await asyncio.wait_for(differentiate_with_timeout(), timeout=25)
                                
                                # Track differentiation result if tracker provided
                                if tracker and company_name:
                                    extracted_name = result.get('extracted_company_name', 'Unknown')
                                    are_same = "Same entity" if differentiation_result.are_same_entity else "Different entity"
                                    tracker.add_data_entry(
                                        company_name=company_name,
                                        operation="Entity Differentiation",
                                        status="Completed",
                                        details=f"Differentiation result for {extracted_name}: {are_same} ({differentiation_result.confidence_level})",
                                        url=search_result_metadata.get('url', '')
                                    )
                                    
                            except (asyncio.TimeoutError, Exception) as e:
                                logger.warning(f"Error in entity differentiation: {e}")
                                # Create basic differentiation result as fallback
                                differentiation_result = EnhancedEntityDifferentiation(
                                    are_same_entity=False,
                                    confidence_level="Low",
                                    explanation=f"Differentiation error: {str(e)}",
                                    critical_conflicts=[],
                                    high_conflicts=[],
                                    potential_relationship=None,
                                    relationship_evidence=None
                                )
                                
                                # Track error if tracker provided
                                if tracker and company_name:
                                    extracted_name = result.get('extracted_company_name', 'Unknown')
                                    tracker.add_data_entry(
                                        company_name=company_name,
                                        operation="Entity Differentiation",
                                        status="Error",
                                        details=f"Error differentiating {extracted_name}: {str(e)}",
                                        url=search_result_metadata.get('url', '')
                                    )
                            
                            # Use strict validation with timeout
                            try:
                                # Apply timeout to validation
                                async def validate_with_timeout():
                                    return await strict_semantic_validation_v4(
                                        target_company_name=cleaned_search_query_name,
                                        target_company_attributes=target_company_attributes,
                                        search_result=search_result_metadata,
                                        entity_differentiation_result=differentiation_result,
                                        negative_examples=negative_examples_collection.negative_examples if negative_examples_collection else []
                                    )
                                
                                validation_result = await asyncio.wait_for(validate_with_timeout(), timeout=25)
                                
                                # Track validation result if tracker provided
                                if tracker and company_name:
                                    extracted_name = result.get('extracted_company_name', 'Unknown')
                                    is_valid = "Valid match" if validation_result.is_valid_match else "Invalid match"
                                    tracker.add_data_entry(
                                        company_name=company_name,
                                        operation="Semantic Validation",
                                        status="Completed",
                                        details=f"Validation result for {extracted_name}: {is_valid} ({validation_result.confidence_level})",
                                        url=search_result_metadata.get('url', '')
                                    )
                                    
                            except (asyncio.TimeoutError, Exception) as e:
                                logger.warning(f"Error in semantic validation: {e}")
                                # Create basic validation result as fallback
                                validation_result = StrictSemanticMatchResult(
                                    is_valid_match=False,
                                    confidence_level="Low",
                                    match_reasoning=f"Validation error: {str(e)}",
                                    is_related_entity=False,
                                    relationship_type=None,
                                    detected_critical_conflicts=[],
                                    negative_example_matches=[],
                                    recommended_fields=[],
                                    warning_message="Error occurred during validation"
                                )
                                
                                # Track error if tracker provided
                                if tracker and company_name:
                                    extracted_name = result.get('extracted_company_name', 'Unknown')
                                    tracker.add_data_entry(
                                        company_name=company_name,
                                        operation="Semantic Validation",
                                        status="Error",
                                        details=f"Error validating {extracted_name}: {str(e)}",
                                        url=search_result_metadata.get('url', '')
                                    )
                            
                            return idx, validation_result
                    except Exception as e:
                        logger.error(f"Error in verification task: {e}")
                        # Return basic validation result for error case
                        
                        # Track error if tracker provided
                        if tracker and company_name:
                            extracted_name = result.get('extracted_company_name', 'Unknown')
                            tracker.add_data_entry(
                                company_name=company_name,
                                operation="Verification Task",
                                status="Failed",
                                details=f"Task error for {extracted_name}: {str(e)}",
                                url=search_result_metadata.get('url', '')
                            )
                            
                        return idx, StrictSemanticMatchResult(
                            is_valid_match=False,
                            confidence_level="Low",
                            match_reasoning=f"Task error: {str(e)}",
                            is_related_entity=False,
                            relationship_type=None,
                            detected_critical_conflicts=[],
                            negative_example_matches=[],
                            recommended_fields=[],
                            warning_message="Exception in verification task"
                        )
                
                verification_tasks.append(verify_non_exact_match(idx, result))
            
            # Execute verification tasks with error handling
            try:
                # Execute all verification tasks with overall timeout
                verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
                
                # Process results
                for result in verification_results:
                    if isinstance(result, Exception):
                        logger.error(f"Verification task failed: {result}")
                        
                        # Track error if tracker provided
                        if tracker and company_name:
                            tracker.add_data_entry(
                                company_name=company_name,
                                operation="Verification Task",
                                status="Exception",
                                details=f"Task failed with exception: {str(result)}"
                            )
                            
                        continue
                        
                    idx, validation_result = result
                    company_semantic_validation[idx] = validation_result
                    
                # Track summary if tracker provided
                if tracker and company_name:
                    valid_count = sum(1 for v in company_semantic_validation.values() if v.is_valid_match)
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation="Semantic Verification",
                        status="Completed",
                        details=f"Verification complete: {valid_count} valid matches out of {len(company_semantic_validation)}"
                    )
                    
            except Exception as e:
                logger.error(f"Error gathering verification results: {e}")
                
                # Track error if tracker provided
                if tracker and company_name:
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation="Semantic Verification",
                        status="Error",
                        details=f"Error gathering verification results: {str(e)}"
                    )
        
        return company_semantic_validation
        
    except Exception as e:
        logger.error(f"Error in semantic verification for {company_name}: {e}")
        
        # Track error if tracker provided
        if tracker and company_name:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Semantic Verification",
                status="Failed",
                details=f"Error in semantic verification: {str(e)}"
            )
            
        return {}

async def perform_field_search(company_name, row_data, cleaned_search_query_name, tracker=None):
    """
    Perform field-specific searches in parallel.
    Fixed to handle direct model object returns and optional tracker parameter.
    """
    try:
        # Review row data for fields needing search
        row_review_result = await review_data_row(
            company_name=company_name,
            row_data=row_data,
            sem=asyncio.Semaphore(5)
        )
        
        # Track row review if tracker provided
        if tracker and company_name:
            fields_count = len(row_review_result.fields_to_review)
            needs_search = "Yes" if row_review_result.needs_additional_search else "No"
            tracker.add_data_entry(
                company_name=company_name,
                operation="Review Data Row",
                status="Completed",
                details=f"Found {fields_count} fields requiring additional search. Needs search: {needs_search}"
            )
        
        # If no fields need search, return empty dict
        if not row_review_result.needs_additional_search:
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Field Search",
                    status="Skipped",
                    details="No fields need additional search"
                )
            return {}
        
        # Generate search queries for each field in parallel
        query_tasks = []
        query_field_names = []
        
        for field_name in row_review_result.fields_to_review:
            query_field_names.append(field_name)
            
            # Generate search query
            query_tasks.append(suggested_query_agent.run(
                user_prompt=f"Company Name: '{company_name}', Field Name: '{field_name}', Field Description: '{COLUMN_TO_MODEL[field_name].__doc__ if field_name in COLUMN_TO_MODEL else ''}'"
            ))
        
        # Track query generation if tracker provided
        if tracker and company_name:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Generate Search Queries",
                status="Started",
                details=f"Generating search queries for {len(query_field_names)} fields"
            )
        
        # Execute all query generation tasks in parallel
        query_results = await asyncio.gather(*query_tasks)
        
        # Process search query results
        fields_for_search = {}
        for i, field_name in enumerate(query_field_names):
            search_query = query_results[i].data.search_query if query_results[i].data else f"{company_name} {field_name}"
            fields_for_search[field_name] = search_query
            
            # Track generated query if tracker provided
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Generate Search Query",
                    status="Completed",
                    details=f"Field: {field_name}, Query: {search_query}"
                )
        
        # Store fields needing search
        field_search_data = {
            "row_index": 0,  # Just use 0 since we're processing one row
            "original_row_data": row_data,
            "field_and_search_query": fields_for_search
        }
        
        # Track search start if tracker provided
        if tracker and company_name:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Field Searches",
                status="Started",
                details=f"Executing searches for {len(fields_for_search)} fields"
            )
        
        # Perform the searches for all fields in parallel
        search_tasks = []
        search_field_names = []
        
        for field_name, search_query in fields_for_search.items():
            search_field_names.append(field_name)
            
            # If tracker provided, use execute_search_with_query_tracking for better tracking
            if tracker and company_name:
                search_task = execute_search_with_query_tracking(
                    search_query,
                    max_results=5,
                    tracker=tracker,
                    company_name=company_name,
                    use_cache=st.session_state.get('use_cache', True),
                    force_refresh=st.session_state.get('force_refresh', False)
                )
            else:
                search_task = process_web_search_results(search_query)
                
            search_tasks.append(search_task)
        
        # Execute all search tasks in parallel
        search_results = await asyncio.gather(*search_tasks)
        
        # Track search completion if tracker provided
        if tracker and company_name:
            success_count = sum(1 for result in search_results if result and hasattr(result, 'search_results') and result.search_results)
            tracker.add_data_entry(
                company_name=company_name,
                operation="Field Searches",
                status="Completed",
                details=f"Completed searches: {success_count} successful out of {len(search_tasks)}"
            )
        
        # Process search results
        field_data_extraction = {}
        
        for i, field_name in enumerate(search_field_names):
            search_result = search_results[i]
            
            # Track field processing if tracker provided
            if tracker and company_name:
                result_count = len(search_result.search_results) if search_result and hasattr(search_result, 'search_results') else 0
                tracker.add_data_entry(
                    company_name=company_name,
                    operation=f"Process Field: {field_name}",
                    status="Started",
                    details=f"Processing {result_count} search results for field {field_name}"
                )
            
            if search_result and hasattr(search_result, 'search_results') and search_result.search_results:
                field_data_list = []
                
                # Process each search result in parallel
                extraction_tasks = []
                
                for item in search_result.search_results:
                    # Create task for content and field data extraction
                    extraction_tasks.append(process_content(
                        item.result_content,
                        item.result_title,
                        item.result_url,
                        field_name=field_name,
                        search_query=item.search_query,
                        tracker=tracker,
                        company_name=company_name
                    ))
                
                # Execute all extraction tasks in parallel
                extraction_results = await asyncio.gather(*extraction_tasks)
                
                # Process extraction results
                for result in extraction_results:
                    if result.get('field_data'):
                        field_data_list.append({
                            'data': result['field_data'],
                            'source': result.get('search_result_metadata', {}).get('url', 'Unknown'),
                            'confidence': "High"  # Default categorical confidence
                        })
                        
                        # Track field data extraction if tracker provided
                        if tracker and company_name:
                            tracker.log_extraction(
                                company_name=company_name,
                                field_name=field_name,
                                extracted_value=result['field_data'],
                                success=True,
                                source_url=result.get('search_result_metadata', {}).get('url', 'Unknown')
                            )
                
                if field_data_list:
                    field_data_extraction[field_name] = field_data_list
                    
                    # Track success if tracker provided
                    if tracker and company_name:
                        tracker.add_data_entry(
                            company_name=company_name,
                            operation=f"Process Field: {field_name}",
                            status="Completed",
                            details=f"Extracted {len(field_data_list)} field data items"
                        )
                else:
                    # Track no data if tracker provided
                    if tracker and company_name:
                        tracker.add_data_entry(
                            company_name=company_name,
                            operation=f"Process Field: {field_name}",
                            status="No Data",
                            details=f"No field data extracted for {field_name}"
                        )
            else:
                # Track no results if tracker provided
                if tracker and company_name:
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation=f"Process Field: {field_name}",
                        status="No Results",
                        details=f"No search results found for field {field_name}"
                    )
        
        # Track final field data extraction summary if tracker provided
        if tracker and company_name:
            if field_data_extraction:
                fields_with_data = len(field_data_extraction)
                total_items = sum(len(items) for items in field_data_extraction.values())
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Field Data Extraction",
                    status="Completed",
                    details=f"Extracted data for {fields_with_data} fields with {total_items} total items"
                )
            else:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Field Data Extraction",
                    status="No Data",
                    details="No field data extracted for any field"
                )
        
        return field_data_extraction
        
    except Exception as e:
        logger.error(f"Error in field search for {company_name}: {e}")
        
        # Track error if tracker provided
        if tracker and company_name:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Field Search",
                status="Error",
                details=f"Error in field search: {str(e)}"
            )
            
        return {}

#########################################
# Field-to-Model Mappings
#########################################

# Mapping of column names to their Pydantic models
COLUMN_TO_MODEL = {
    "Website": str,
    "Products / Services": ProductServiceDetails,
    "Publicly Announced Partnerships": PartnershipDetails,
    "Business Model(s)": BusinessModelDetails,
    "Lab or Proprietary Data Generation?": LabDataGeneration,
    "Drug Pipeline?": DrugPipeline,
    "Organization Type": OrganizationType,
    "HQ Location": HQLocations,
    "Relevant Segments": RelevantSegmentDetails,
    "Funding Stage": FundingStageBooleans,
    "RoM Estimated Funding or Market Cap for Public Companies": FundingMarketCapBooleans,
    "Year Founded": YearFoundedDetails,
    "Advisor / Board Members / Key People": EnhancedKeyPeopleDetails,  # Updated to enhanced version
    "Funding Rounds": FundingRoundList,
    "Investors": InvestorList,
    "Female Co-Founder?": FemaleCoFounder,
    "Watchlist": WatchlistItem,
    "Scientific Domain": ScientificDomainDetails,  # New field
    "Competitive Landscape": CompetitiveLandscape,  # New field
}

# Mapping of column names to their extraction agent functions
# Modified COLUMN_TO_AGENT_FUNCTION mapping with proper error handling
COLUMN_TO_AGENT_FUNCTION = {
    "Products / Services": lambda content: handle_field_extraction(extract_enhanced_product_service_details, content),  # Enhanced
    "Publicly Announced Partnerships": lambda content: handle_field_extraction(extract_partnership_details, content),
    "Business Model(s)": lambda content: handle_field_extraction(extract_business_model_details, content),  # Updated
    "Lab or Proprietary Data Generation?": lambda content: handle_field_extraction(extract_lab_data_generation, content),
    "Drug Pipeline?": lambda content: handle_field_extraction(extract_drug_pipeline, content),
    "Organization Type": lambda content: handle_field_extraction(extract_organization_type, content),
    "HQ Location": lambda content: handle_field_extraction(extract_hq_locations, content),
    "Relevant Segments": lambda content: handle_field_extraction(extract_relevant_segments, content),
    "Funding Stage": lambda content: handle_field_extraction(extract_funding_stage, content),
    "RoM Estimated Funding or Market Cap for Public Companies": lambda content: handle_field_extraction(extract_funding_market_cap, content),
    "Year Founded": lambda content: handle_field_extraction(extract_year_founded, content),
    "Advisor / Board Members / Key People": lambda content: handle_field_extraction(extract_enhanced_key_people_details, content),  # Enhanced
    "Funding Rounds": lambda content: handle_field_extraction(extract_funding_rounds, content),
    "Investors": lambda content: handle_field_extraction(extract_investor_details, content),
    "Female Co-Founder?": lambda content: handle_field_extraction(extract_female_co_founder, content),
    "Watchlist": lambda content: handle_field_extraction(extract_watchlist_item, content),
    "Scientific Domain": lambda content: handle_field_extraction(extract_scientific_domain_details, content),  # New field
    "Competitive Landscape": lambda content: handle_field_extraction(extract_competitive_landscape, content),  # New field
}


# 1. Enhanced function to display search queries with better debugging
def display_search_queries_summary(cleaned_search_query_name, results_with_metadata):
    """
    Displays a summary of the search queries used and the number of results per query.
    Enhanced with better error handling and debugging.
    """
    if not results_with_metadata:
        st.info("No search results available.")
        return
    
    # Create a debugging expander
    with st.expander("Search Query Debugging Info", expanded=False):
        st.write("First result structure:")
        if results_with_metadata:
            # Show keys of first result
            st.write("Result keys:", list(results_with_metadata[0].keys()))
            
            # Show metadata structure if it exists
            if 'search_result_metadata' in results_with_metadata[0]:
                st.write("Metadata keys:", list(results_with_metadata[0]['search_result_metadata'].keys()))
            
            # Show full structure (limited)
            st.json(results_with_metadata[0])
    
    # Extract search queries and count results with enhanced detection
    query_counts = {}
    queries_found = False
    
    for result in results_with_metadata:
        # Try multiple possible locations for search_query
        query = None
        
        # Location 1: In search_result_metadata (most common)
        if result.get('search_result_metadata', {}).get('search_query'):
            query = result['search_result_metadata']['search_query']
            queries_found = True
        
        # Location 2: Directly in result (sometimes happens)
        elif result.get('search_query'):
            query = result['search_query']
            queries_found = True
            
        # Location 3: In search_result_metadata.content (rare)
        elif isinstance(result.get('search_result_metadata', {}).get('content'), dict) and result['search_result_metadata']['content'].get('search_query'):
            query = result['search_result_metadata']['content']['search_query']
            queries_found = True
            
        if query:
            if query in query_counts:
                query_counts[query] += 1
            else:
                query_counts[query] = 1
    
    # Display summary
    if query_counts:
        st.subheader(f"Search Queries Used for {cleaned_search_query_name.title()}")
        
        # Create DataFrame for better display
        query_data = []
        for query, count in query_counts.items():
            query_data.append({
                "Search Query": query,
                "Number of Results": count
            })
        
        query_df = pd.DataFrame(query_data)
        st.dataframe(query_df)
        
        total_results = sum(query_counts.values())
        st.success(f"Total: {len(query_counts)} search queries used to find {total_results} results")
    elif not queries_found:
        st.warning("No search queries tracked in the results. This may be because the 'search_query' field is missing in the metadata.")
        
        # Offer a solution
        st.info("To fix this issue, check that search queries are being properly stored in the search results metadata.")
        
        # Add a manual solution
        st.subheader("Manual Query Tracking")
        manual_queries = [
            "Company information",
            "Funding rounds",
            "Founders and team",
            "Technology and products"
        ]
        manual_query_df = pd.DataFrame({
            "Search Query": manual_queries,
            "Number of Results": ["N/A"] * len(manual_queries)
        })
        st.write("Showing example search queries (actual queries not tracked):")
        st.dataframe(manual_query_df)

async def execute_search_with_query_tracking(query, max_results=5, search_depth="advanced", 
                                            tracker=None, company_name=None,
                                            use_cache=True, force_refresh=False,
                                            cache_type=None):
    """
    Execute a search with query tracking and comprehensive error handling.
    """
    start_time = time.time()
    
    # Log search query execution if tracker provided
    if tracker and company_name:
        tracker.log_search_query(company_name, query, start_time)
    
    try:
        # Check cache first if enabled and not forcing refresh
        if use_cache and not force_refresh:
            cached_results = load_from_cache(query, cache_type or "general")
            if cached_results:
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                logger.info(f"Using cached results for query: {query}")
                
                # Explicitly add query information to cached results
                for result in cached_results:
                    # Store at top level
                    result['search_query'] = query
                    
                    # Ensure it's also in metadata if that exists or will be created
                    if 'metadata' not in result:
                        result['metadata'] = {}
                    result['metadata']['search_query'] = query
                
                # Log cached results if tracker provided
                if tracker and company_name:
                    tracker.add_data_entry(
                        company_name=company_name,
                        operation="Search Query",
                        status="Cached",
                        details=f"Using {len(cached_results)} cached results for query: {query}",
                        search_query=query,
                        result_count=len(cached_results),
                        duration=elapsed_time
                    )
                    
                    # Log each result if tracker provided
                    for result in cached_results:
                        tracker.log_search_result(
                            company_name=company_name,
                            result=result,
                            query=query
                        )
                
                return cached_results
        
        # Perform actual search if no cache or cache disabled
        async with sem:
            try:
                search_results_raw = await tavily_client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=True,
                    include_raw_content=True
                )
            except Exception as api_error:
                # Check if this is a credit exhaustion error
                if handle_tavily_credit_exhaustion(api_error, company_name, tracker):
                    # If credits are exhausted, enable cache-only mode and try to use cache
                    st.session_state.use_cache = True
                    st.session_state.force_refresh = False
                    
                    # Try to load from cache again as a fallback
                    cached_results = load_from_cache(query, cache_type or "general")
                    if cached_results:
                        if tracker and company_name:
                            tracker.add_data_entry(
                                company_name=company_name,
                                operation="Search Query",
                                status="Using Cache (API Credits Exhausted)",
                                details=f"Using cached results due to API credit exhaustion",
                                search_query=query
                            )
                        return cached_results
                    
                    # If no cache available, return empty results
                    return []
                else:
                    # For other types of API errors, rethrow
                    raise
            
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
            
        if not search_results_raw or not search_results_raw.get('results'):
            logger.warning(f"No results found for query: {query}")
            
            # Log empty results if tracker provided
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Search Query",
                    status="No Results",
                    details=f"No results found for query: {query}",
                    search_query=query,
                    duration=elapsed_time
                )
            return []
            
        results = search_results_raw.get('results', [])
        
        # Explicitly add query information at multiple levels to ensure it's preserved
        for result in results:
            # Store at top level
            result['search_query'] = query
            
            # Ensure it's also in metadata if that exists or will be created
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata']['search_query'] = query
            
            # Log each result if tracker provided
            if tracker and company_name:
                tracker.log_search_result(
                    company_name=company_name,
                    result=result,
                    query=query
                )
        
        # Save to cache if enabled
        if use_cache:
            save_to_cache(query, cache_type or "general", results, max_results, search_depth)
        
        # Log successful search if tracker provided
        if tracker and company_name:
            tracker.add_data_entry(
                company_name=company_name,
                operation="Search Query",
                status="Completed",
                details=f"Found {len(results)} results for query: {query}",
                search_query=query,
                result_count=len(results),
                duration=elapsed_time
            )
            
        return results
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error in search for '{query}': {e}")
        
        # Check again if this is a credit exhaustion error
        if not handle_tavily_credit_exhaustion(e, company_name, tracker):
            # For other types of errors, log as usual
            if tracker and company_name:
                tracker.add_data_entry(
                    company_name=company_name,
                    operation="Search Query",
                    status="Error",
                    details=f"Error executing search: {str(e)}",
                    search_query=query,
                    duration=elapsed_time
                )
        
        return []

def display_processing_status(processing_companies):
    """
    Displays the current processing status of companies without using nested expanders.
    
    Args:
        processing_companies: Dictionary of company names and their processing status
    """
    if not processing_companies:
        return
    
    st.subheader("Company Processing Status")
    
    # Create a DataFrame for better display
    status_data = []
    for company, status in processing_companies.items():
        if status == "completed":
            status_icon = "✅"
            status_text = "Completed"
        elif status == "error":
            status_icon = "❌"
            status_text = "Error"
        elif status == "processing":
            status_icon = "⏳"
            status_text = "Processing"
        else:
            status_icon = "🔄"
            status_text = "Pending"
        
        status_data.append({
            "Company": company,
            "Status": f"{status_icon} {status_text}"
        })
    
    # Display as a table
    if status_data:
        status_df = pd.DataFrame(status_data)
        st.table(status_df)


def main():
    """Main Streamlit application with safe async execution."""
    st.set_page_config(
        page_title="AI Company Research Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with st.sidebar:
        # Conditional API status check
        if 'api_status' not in st.session_state or st.button("Check API Status"):
            api_status_container = st.empty()
            with st.spinner("Checking Tavily API status..."):
                is_available, has_credits, status_message = safe_async_run(check_tavily_api_status())
            
            # Store the status check results in session state
            st.session_state.api_status = {
                'is_available': is_available,
                'has_credits': has_credits,
                'status_message': status_message,
                'last_checked': datetime.now().isoformat()
            }
            
            if not is_available:
                api_status_container.error(f"Tavily API connection issue: {status_message}")
                st.warning("Search functionality will be limited to cached results only.")
                st.session_state.use_cache = True
                st.session_state.force_refresh = False
            elif not has_credits:
                api_status_container.error("⚠️ Tavily API credits exhausted")
                st.warning(
                    "Your Tavily API credits have been depleted. Please add more credits "
                    "to your Tavily AI account to continue using the search functionality."
                )
                st.info(
                    "To add more credits to your Tavily account:\n"
                    "1. Visit [Tavily Pricing Page](https://tavily.com/pricing)\n"
                    "2. Sign in to your account\n"
                    "3. Select an appropriate plan or add more credits to your existing plan"
                )
                
                # Automatically enable cache-only mode
                st.success("Automatically switched to cache-only mode. You can continue working with previously cached search results.")
                st.session_state.use_cache = True
                st.session_state.force_refresh = False
            else:
                api_status_container.success("Tavily API connection established successfully.")
        else:
            # Display the status from session state
            last_checked = st.session_state.api_status.get('last_checked', 'unknown')
            if last_checked != 'unknown':
                try:
                    dt = datetime.fromisoformat(last_checked)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    formatted_time = last_checked
            else:
                formatted_time = 'unknown'
                
            st.info(f"API Status (last checked: {formatted_time})")
            st.write(f"API Available: {'✅' if st.session_state.api_status.get('is_available', False) else '❌'}")
            st.write(f"API Credits: {'✅ Available' if st.session_state.api_status.get('has_credits', False) else '❌ Exhausted'}")
            
            # Add a button to manually check again
            st.button("Recheck API Status")

    
    # Make sure we have nest_asyncio applied at startup
    if 'asyncio_initialized' not in st.session_state:
        import nest_asyncio
        nest_asyncio.apply()
        st.session_state.asyncio_initialized = True

    if 'use_cache' not in st.session_state:
        st.session_state.use_cache = True
    
    if 'force_refresh' not in st.session_state:
        st.session_state.force_refresh = False
    
    # Add cache controls to sidebar
    add_cache_controls_to_sidebar()
    
    # Add cache usage stats to sidebar
    add_cache_usage_stats_to_sidebar()
    
    # Initialize entity database if not exists
    if 'entity_database' not in st.session_state:
        st.session_state.entity_database = EntityDatabase()
    
    # Initialize session state variables if not exists
    if 'search_results_all_companies' not in st.session_state:
        st.session_state.search_results_all_companies = {}
    
    if 'selected_company' not in st.session_state:
        st.session_state.selected_company = None
    
    if 'initial_processing_complete' not in st.session_state:
        st.session_state.initial_processing_complete = False
    
    st.title("AI Company Research Assistant")
    st.markdown("""
    This tool helps research companies by using AI to extract structured information from web searches.
    Upload an Excel file with company names in the first column, or enter company names manually.
    """)
    
    st.success("✅ Using enhanced entity differentiation and validation")
    st.info("""
    This tool helps research companies by using AI to extract structured information from web searches.
    Upload an Excel file with company names in the first column, or enter company names manually.
    
    Enhanced v4 features include:
    - More accurate entity differentiation
    - Improved semantic validation
    - Negative example identification
    - Investment-focused query generation
    - Priority-weighted multi-source search
    
    Displays a summary of the search queries used and the number of results per query.
    Enhanced with better error handling and debugging.
    """)
    
    with st.expander("Example Output Table"):
        st.markdown("""
        ## Example Output Table:
        Below is an example of the enhanced output after implementing these changes:
        
        | Field Name              | Value                                      | Sources                        | Confidence |
        |-------------------------|--------------------------------------------|--------------------------------|------------|
        | Company Name            | Acme Biotech                               | crunchbase.com, company website| High       |
        | Products / Services     | GeneSynth (AI-driven gene synthesis platform)| company website, techcrunch.com| High     |
        | Business Model(s)       | SaaS, Licensing, Research Collaboration    | company website, forbes.com    | High       |
        | Scientific Domain       | Biotechnology (Synthetic Biology, Gene Therapy)| company website, nature.com | Very High  |
        | HQ Location             | Cambridge, MA, USA                         | crunchbase.com, linkedin.com   | High       |
        | Funding Stage           | Series B                                   | crunchbase.com, pitchbook.com  | High       |
        | RoM Estimated Funding   | Significant (low 100s of $M)               | pitchbook.com                  | Medium     |
        | Year Founded            | 2018                                       | crunchbase.com, linkedin.com   | High       |
        | Key People              | Jane Doe (Founder, CEO)                    | company website, forbes.com    | High       |
        |                         | John Smith (Executive, CTO)                | company website, linkedin.com  | High       |
        |                         | Sarah Johnson (Board Member, Investor)     | crunchbase.com                 | Medium     |
        | Funding Rounds          | May 1, 2022 - $45,000,000 - Series B       | crunchbase.com, techcrunch.com | High       |
        |                         | June 15, 2020 - $12,000,000 - Series A     | crunchbase.com                 | High       |
        |                         | Total Funding: $60,000,000                 | crunchbase.com                 | High       |
        | Investors               | Breakthrough Ventures - Series B Lead      | crunchbase.com, techcrunch.com | High       |
        |                         | Biotech Capital - Series A Lead            | crunchbase.com                 | High       |
        | Competitive Landscape   | Competitors: GeneWorks, SynthBio, BioGenesis | pitchbook.com, industry reports | Medium  |
        """)
        
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Single Company", "Batch Processing", "Results"])
    
    with tab1:
        st.header("Research a Single Company")
        
        # Restore previous input if available
        previous_company = st.session_state.get('single_company_name', '')
        previous_url = st.session_state.get('single_company_url', '')
        
        # Input for single company name and optional URL
        company_name = st.text_input("Enter Company Name:", value=previous_company, key="single_company_name_input")
        company_url = st.text_input("Company Website URL (optional):", value=previous_url, key="single_company_url_input")
        
        # Update session state when inputs change
        if company_name != previous_company:
            st.session_state.single_company_name = company_name
        
        if company_url != previous_url:
            st.session_state.single_company_url = company_url
                
        if st.button("Research Company") and company_name:
            try:
                company_urls = [company_url] if company_url else None
                                
                # Create containers for live updates
                result_status = st.empty()
                search_status = st.empty()
                result_data = st.empty()
                
                # Execute search for single company
                with st.spinner(f"Researching {company_name}..."):
                    # Create containers for live updates
                    query_container = search_status.container()
                    search_results_container = search_status.container()
                    company_data_container = search_status.container()
                    
                    # Pass cache control parameters from session state
                    search_results = safe_async_run(
                        safe_enhanced_search_company_summary_v4(
                            company_name=company_name,
                            company_urls=company_urls,
                            use_test_data=False,
                            NUMBER_OF_SEARCH_RESULTS=7,
                            original_input=company_name,
                            query_container=query_container,
                            search_results_container=search_results_container,
                            company_data_container=company_data_container,
                            use_cache=st.session_state.use_cache,
                            force_refresh=st.session_state.force_refresh
                        )
                    )
                    
                    # Updated unpacking to include all 7 values
                    (grouped_results_dict, 
                     cleaned_search_query_name, 
                     results_with_metadata, 
                     selection_results, 
                     non_exact_match_results_metadata, 
                     company_features,
                     negative_examples_collection) = search_results
                    
                    # Store in session state for later use
                    st.session_state.search_results_all_companies[company_name] = {
                        "grouped_results_dict": grouped_results_dict,
                        "cleaned_search_query_name": cleaned_search_query_name,
                        "results_with_metadata": results_with_metadata,
                        "selection_results": selection_results,
                        "non_exact_match_results_metadata": non_exact_match_results_metadata,
                        "company_features": company_features,
                        "company_url": [company_url] if company_url else None,
                        "v4_processed": True,
                        "negative_examples_collection": negative_examples_collection  # Add this to session state
                    }
                    st.session_state.selected_company = company_name
                
                # Clear the status containers once search is complete
                search_status.empty()
                
                # Prepare data for aggregation
                company_data_for_table_exact = []
                for entity_name, result_list in grouped_results_dict.items():
                    for result_dict in result_list:
                        if result_dict.get('company_data'):
                            company_data_for_table_exact.append(result_dict['company_data'])
                
                search_result_metadata_list = [
                    result_dict['search_result_metadata'] 
                    for result_dict in results_with_metadata 
                    if result_dict.get('search_result_metadata')
                ]
                
                # Display results - using safe_async_run
                with result_data:
                    safe_async_run(display_table_results(company_data_for_table_exact, cleaned_search_query_name, search_result_metadata_list))
                
                result_status.success(f"Research completed for {company_name} with Enhanced V4 processing! View details in the Results tab.")
            
            except Exception as e:
                st.error(f"Error processing company: {str(e)}")
                logger.error(f"Error in single company research: {traceback.format_exc()}")
    
    with tab2:
        # Use the updated batch processing implementation
        updated_tab2_implementation_v3()
        
    with tab3:
        st.header("Detailed Results")
        
        # Display warning if no results available
        if not st.session_state.search_results_all_companies:
            st.info("No company results available yet. Please research a company first in the Single Company or Batch Processing tab.")
            return
        
        # Select company to display results for
        company_options = list(st.session_state.search_results_all_companies.keys())
        
        if company_options:
            selected_company = st.selectbox(
                "Select Company to View Results:",
                options=company_options,
                index=company_options.index(st.session_state.selected_company) if st.session_state.selected_company in company_options else 0,
                key="results_company_selector"
            )
            
            # Update selected company in session state
            st.session_state.selected_company = selected_company
            
            # Check if this company was processed with V4
            v4_processed = st.session_state.search_results_all_companies[selected_company].get('v4_processed', False)
            if v4_processed:
                st.info("This company was processed with Enhanced V4 entity differentiation.")
            
            # Display aggregated results if available
            if 'all_companies_aggregated_data' in st.session_state and 'final_df' in st.session_state:
                st.subheader("Aggregated Results")
                safe_display_dataframe(st.session_state.final_df, use_gradient=True)
                
                # Add download button for aggregated results
                if 'final_df' in st.session_state:
                    try:
                        excel_buffer = io.BytesIO()
                        st.session_state.final_df.to_excel(excel_buffer, index=False)
                        excel_buffer.seek(0)
                        
                        st.download_button(
                            label="Download Aggregated Results as Excel",
                            data=excel_buffer,
                            file_name="company_research_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    except Exception as e:
                        st.error(f"Error preparing Excel download: {str(e)}")
                        
                        # Offer CSV as a fallback
                        csv_buffer = io.StringIO()
                        st.session_state.final_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        st.download_button(
                            label="Download Aggregated Results as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="company_research_results.csv",
                            mime="text/csv",
                        )
            
            # Display V4-specific results if available and processed with V4
            if v4_processed and 'negative_examples_collection' in st.session_state.search_results_all_companies[selected_company]:
                neg_examples = st.session_state.search_results_all_companies[selected_company]['negative_examples_collection']
                if neg_examples and hasattr(neg_examples, 'negative_examples') and neg_examples.negative_examples:
                    with st.expander("Negative Examples Identified", expanded=False):
                        st.write(f"Found {len(neg_examples.negative_examples)} entities that are definitely NOT {selected_company}:")
                        for i, example in enumerate(neg_examples.negative_examples):
                            st.markdown(f"**Entity {i+1}:** {example.entity_name}")
                            st.write(f"Confidence: {example.confidence_score}")
                            st.write(f"Key differentiators: {', '.join(example.key_differentiators[:3])}")
                            st.divider()
            
            # Display interactive results for the selected company
            if selected_company in st.session_state.search_results_all_companies:
                st.subheader(f"Detailed Results for {selected_company}")
                selection_results = st.session_state.search_results_all_companies[selected_company].get("selection_results", [])
                
                # Choose the appropriate display function based on V4 processing
                if v4_processed:
                    # Option 1: Only call it once - remove this direct call
                    # safe_async_run(display_interactive_non_exact_matches_v4(selected_company))
                    
                    # Option 2: Or modify display_interactive_results to not call it again
                    # Add a parameter to communicate this
                    safe_async_run(display_interactive_results(selected_company, selection_results, skip_non_exact=True))
                else:
                    safe_async_run(display_interactive_results(selected_company, selection_results))

            else:
                st.info("No detailed results available for this company yet.")
        else:
            st.info("No company results available. Please research a company first.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
