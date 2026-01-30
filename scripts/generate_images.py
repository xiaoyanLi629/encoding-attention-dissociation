#!/usr/bin/env python3
"""
Image Generator for Chomsky Hierarchy Paper
============================================

This script generates academic illustrations using either:
- OpenAI DALL-E 3 API
- Google Gemini/Imagen API

Usage:
    python generate_images.py --model dalle3
    python generate_images.py --model gemini
    python generate_images.py --model both

API Keys should be stored in api_keys.txt in the same directory.

Required packages:
    pip install openai requests
    pip install google-genai  # For Gemini image generation
"""

import argparse
import requests
import time
import base64
from pathlib import Path

# Try to import required libraries
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try google-genai (newer unified SDK with image generation support)
GEMINI_AVAILABLE = False
GEMINI_SDK_TYPE = None

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
    GEMINI_SDK_TYPE = "google-genai"
except ImportError:
    pass

# Fallback: try google-generativeai (older SDK, limited image gen support)
if not GEMINI_AVAILABLE:
    try:
        import google.generativeai as genai_legacy
        # Note: google-generativeai has limited image generation support
        # It's primarily for text/multimodal understanding, not image creation
        GEMINI_AVAILABLE = True
        GEMINI_SDK_TYPE = "google-generativeai"
    except ImportError:
        pass


# ============================================================
# Configuration
# ============================================================

def load_api_keys():
    """Load API keys from api_keys.txt file."""
    keys = {}
    keys_file = Path(__file__).parent / "api_keys.txt"
    
    if keys_file.exists():
        with open(keys_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    keys[key.strip()] = value.strip()
    
    return keys

API_KEYS = load_api_keys()
OPENAI_API_KEY = API_KEYS.get('OPENAI_API_KEY', '')
GEMINI_API_KEY = API_KEYS.get('NANOBANANAPRO_API_KEY', '')

# Output directories
BASE_DIR = Path(__file__).parent
DALLE3_DIR = BASE_DIR / "media" / "dalle3"
GEMINI_DIR = BASE_DIR / "media" / "gemini"


# ============================================================
# Image Prompts (from image_prompts.md)
# ============================================================

PROMPTS = {
    "project_overview_intro": {
        "filename": "project_overview_intro.png",
        "prompt": """Create a highly professional, publication-quality scientific illustration for a cognitive neuroscience paper introduction, depicting the concept of "Encoding-Attention Dissociation in Multimodal Brain Processing". 

The figure should be arranged as a clean horizontal layout with three connected sections on a pure white background with very subtle light gray scientific grid pattern.

LEFT SECTION - "Multimodal Input Processing": 
Show three parallel input streams representing Visual (movie frames icon in warm red-orange), Audio (sound waveform icon in soft blue), and Language (text/speech bubble icon in teal green) modalities. Each stream flows through a stylized neural network encoder block. Below each encoder, show encoding strength bars: Visual bar is 2x longer (prominent, ~91% label), Audio and Language bars are similar shorter lengths (~9% combined). Label this section "Feature Extraction & Encoding" with subtitle "Visual Dominance: 91% of brain regions".

CENTER SECTION - "Hierarchical Brain Architecture":
Illustrate a simplified side-view brain silhouette in very light gray outline with three highlighted regions: 
1) Posterior occipital area (Visual Network) in light red with label "Modality-Specific" and "MII=0.56"
2) Lateral temporal area (Auditory) in light blue
3) Medial prefrontal and posterior cingulate (Default Mode Network) highlighted prominently in deep teal/green with glowing effect, labeled "Integration Hub" and "MII=0.74"
Draw gradient arrows flowing from sensory regions toward DMN, suggesting hierarchical processing. Add small network icons showing the Schaefer 7-network parcellation concept.

RIGHT SECTION - "The Dissociation":
Show a key comparison visualization with two pie charts or bar groups side by side:
- Left chart labeled "Encoding Strength": dominated by red (Visual ~48%), with small blue (Audio) and green (Language) segments
- Right chart labeled "Attention Weights": three nearly equal segments (~33% each) in red, blue, and green
Draw a prominent "≠" symbol between them with a dotted circle highlight
Below, add text box: "Apparent Inefficiency → Hidden Optimality"
Add small brain icon with balanced scales symbol representing "Robust Integration"

BOTTOM BANNER:
Three connected boxes showing the causal chain: "C1: Hybrid Architecture" → "C2: Hierarchical Integration" → "C3: Balanced Attention" with arrows between them. Each box has a small representative icon.

VISUAL STYLE:
- Clean, minimalist scientific illustration style suitable for Nature/Science/PNAS journals
- Color scheme: warm red-orange for visual, soft sky blue for audio, teal-green for language, deep blue-green gradient for DMN
- All text in clean sans-serif font (Arial/Helvetica style)
- Subtle drop shadows and gradients for depth, but no excessive 3D effects
- Include small legend box explaining color coding
- Main title at top: "The Encoding-Attention Dissociation in Multimodal Brain Processing"
- Subtitle: "How the Brain Achieves Robust Integration Despite Unequal Feature Strength"

Resolution: 4K quality, 16:9 landscape orientation, 600 DPI suitable for print publication. Professional academic illustration aesthetic matching Springer Nature or Cell Press figure standards. All elements balanced and well-spaced with clear visual hierarchy guiding the reader from left to right."""
    },
    
    "chomsky_hierarchy_nested": {
        "filename": "chomsky_hierarchy_nested.png",
        "prompt": """Create a highly detailed, professional academic diagram illustrating the Chomsky Hierarchy as a nested containment structure. The composition should be centered on a clean, pure white background with subtle light gray grid lines barely visible to suggest mathematical precision. The diagram consists of four concentrically nested rounded rectangles, each representing a level of the hierarchy with consistent padding between each layer. Layer 1 (outermost) Type-0 in deep navy blue represents Recursively Enumerable Languages recognized by Turing Machines with a small infinite tape icon. Layer 2 Type-1 in teal green represents Context-Sensitive Languages recognized by Linear Bounded Automata with a bounded tape icon. Layer 3 Type-2 in warm orange represents Context-Free Languages recognized by Pushdown Automata with a stack symbol icon. Layer 4 (innermost) Type-3 in crimson red represents Regular Languages recognized by Finite Automata with a simple state diagram icon. Each layer should have subtle gradient fill getting lighter toward the center, with primary labels in bold sans-serif font and secondary labels showing the recognizing automaton. Include example languages in mathematical notation for each level. On the right side add a vertical arrow labeled "Increasing Computational Power" with gradient from red to blue. On the left side add a vertical arrow labeled "Increasing Restrictions on Grammar Rules". Overall aesthetic should be clean, minimalist, professional academic illustration similar to Springer or MIT Press computer science textbooks, vector-style graphics with crisp edges, 4K resolution, 16:9 landscape orientation, no decorative elements or 3D effects, pure flat design, color scheme accessible for colorblind viewers."""
    },
    
    "dfa_example": {
        "filename": "dfa_example.png",
        "prompt": """Create a meticulously detailed technical diagram of a Deterministic Finite Automaton (DFA) that recognizes the regular language L = {w in {a,b}* | w contains the substring "abb"}. Pure white background with subtle dot grid pattern in very light gray with 20px spacing to suggest graph paper. The DFA has exactly 4 states arranged horizontally from left to right with consistent spacing between state centers. State q0 is the initial state shown as a circle with white fill with very subtle light blue tint, solid black border, label "q0" centered inside, with a short arrow pointing to it from the left with no origin state indicating initial state, small label "Start" below. State q1 same circle style with label "q1" and annotation "seen 'a'" below. State q2 same circle style with label "q2" and annotation "seen 'ab'" below. State q3 is the accepting final state shown as double circle with very subtle light green tint fill, label "q3" inside, small label "Accept" below, annotation "seen 'abb'" below. All transitions are curved or straight arrows with dark gray color, filled triangle arrow heads, labels positioned at midpoint with small white background rectangle behind text. Transitions: q0 to q1 straight arrow labeled "a", q0 self-loop on top labeled "b", q1 self-loop on top labeled "a", q1 to q2 straight arrow labeled "b", q2 to q1 curved arrow going backward arc above states labeled "a", q2 to q3 straight arrow labeled "b", q3 self-loop on top labeled "a", q3 self-loop on bottom labeled "b". Title at top "DFA for L = {w in {a,b}* | w contains 'abb'}" in bold sans-serif. Subtitle "A 4-state deterministic finite automaton" in regular gray. Legend box at bottom explaining notation. Crisp vector-quality lines, professional technical illustration style, consistent line weights, no decorative elements or gradients, suitable for black-and-white printing."""
    },
    
    "pda_architecture": {
        "filename": "pda_architecture.png",
        "prompt": """Create an extremely detailed educational technical illustration of a Pushdown Automaton (PDA) architecture that recognizes the context-free language L = {a^n b^n | n >= 0}. Clean white background with very subtle blueprint-style grid in light blue. The diagram is divided into three main sections: Input Tape at top, Finite Control in middle-left, and Stack on right side. Input Tape section shows a long horizontal rectangle divided into equal square cells, 10 cells visible with "..." on both ends indicating extension, cell contents left to right showing "a a a a b b b b blank" where blank represents empty, cells containing 'a' have subtle green tint and cells containing 'b' have subtle blue tint, cell labels in monospace font centered. Read Head positioned below the tape pointing up at the first 'b' cell, shown as inverted triangle with rectangular base in bright red fill with darker red border, label "Read Head" below, annotation arrow pointing to current cell "Current Input Symbol: b". Finite Control Unit in center-left shown as large rounded rectangle with light blue fill and darker blue border, label "FINITE CONTROL" at top inside in bold, containing three states in triangle formation: q0 at top as double circle indicating both initial and accepting state with incoming arrow from outside and labels "start/accept", q1 at bottom-left as single circle with annotation "pushing phase", q2 at bottom-right as single circle with annotation "popping phase". Transitions inside control unit with labels in format showing input symbol, stack top, and stack operation. Stack on right side shown as vertical rectangle with solid orange border and very light orange fill, stack contents from bottom showing Z0 as stack bottom marker in darker orange background then four A symbols stacked above with the top A highlighted with yellow glow, empty cells above shown with dashed borders, annotations showing "Stack Top" and "Stack Bottom Marker Z0" and "LIFO Stack" in vertical text. Large curved arrows connecting components labeled "Input Symbol" from Read Head to Finite Control, "Stack Operations" from Finite Control to Stack, "Stack Top Symbol" from Stack back to Finite Control, these arrows thick with gradient colors and motion indicators. Information panel at bottom showing current configuration and language description. Main title "Pushdown Automaton (PDA) Architecture" bold at top center, subtitle "Recognizing the Context-Free Language {a^n b^n | n >= 0}" regular. Clean educational illustration style, consistent color coding blue for control orange for stack green for input, all text highly legible, suitable for textbook or lecture slides."""
    },
    
    "turing_machine": {
        "filename": "turing_machine.png",
        "prompt": """Create a comprehensive highly detailed technical illustration of a Turing Machine showing its complete architecture and demonstrating a computation step. Pure white background with extremely subtle gray texture suggesting paper. Main title block at top "The Turing Machine: A Universal Model of Computation" in elegant serif font bold, subtitle "Proposed by Alan Turing in 1936" in italic, horizontal decorative line below. The Infinite Tape in upper section spans almost entire width, divided into square cells, 15 cells visible with both ends fading into "..." with gradient transparency suggesting infinite extension, each cell has solid dark gray border with alternating very subtle cream tint, cell contents showing symbols "blank blank 1 0 1 1 blank 1 1 0 blank blank" in monospace font bold centered with '1' in dark blue '0' in dark green 'blank' in light gray, tape labels above showing "INFINITE TAPE" in small caps with arrows on both ends indicating infinite extension, index numbers below cells. Read/Write Head positioned below cell containing '1', complex mechanical-looking design with main body as rounded rectangle with top connector to tape cell, sensor window at top showing current symbol, gradient from steel gray to darker gray with metallic appearance and subtle highlights, red indicator light showing active status, labels "R/W HEAD" inside in white text, annotations showing "READ: Currently reading '1'" and "WRITE: Can write any symbol from Gamma" and movement indicators "MOVE LEFT" and "MOVE RIGHT" with arrow icons. Finite State Control in lower left as large rounded rectangle with light blue gradient background and solid medium blue border with subtle drop shadow, title "FINITE STATE CONTROL" bold centered at top, containing six states arranged meaningfully: q0 initial state top-left with incoming arrow, q1 top-right labeled "scan right", q2 bottom-left labeled "carry", q3 bottom-center labeled "return", qa accept state bottom-right as double circle with light green fill, qr reject state far right with red fill, state q1 highlighted with glowing yellow border showing current state, 5-6 representative transitions with labels in format "(read, write, move)". Formal Definition Panel in lower right as rounded rectangle with very light gray background and solid gray border containing the 7-tuple definition M = (Q, Sigma, Gamma, delta, q0, qa, qr) with explanations of each component. Computation Step Illustration at bottom center showing before and after comparison with small tape segments, head positions, and state indicators with arrow between labeled with transition function and explanation below. Professional technical illustration quality, consistent visual language throughout, color coding blue for control orange for tape operations green for accept red for reject, all fonts crisp and legible, suitable for academic publication or high-quality textbook, clean modern aesthetic with subtle depth minimal shadows no excessive 3D effects."""
    },
    
    "parse_tree": {
        "filename": "parse_tree.png",
        "prompt": """Create an exquisitely detailed syntax tree parse tree diagram demonstrating the derivation of the sentence "the clever cat quickly chased the frightened mouse" using a context-free grammar. Pure white background and subtle decorative double line border from edge. Title section with main title "Context-Free Grammar Parse Tree" in elegant serif font bold, subtitle "Syntactic Analysis of: 'the clever cat quickly chased the frightened mouse'" in italic, grammar notation box in top-right corner showing the grammar rules used. Tree structure with root node S for Sentence at top center as large circle with deep blue fill and white text, solid darker blue border, label "S" bold white centered, annotation "Sentence (Start Symbol)" above in small gray text. Level 1 direct children of S: NP node for Noun Phrase in upper-left as circle with teal fill and white text label "NP", VP node for Verb Phrase in upper-right as circle with orange fill and white text label "VP". Level 2 children of left NP: Det node, Adj node, N node each as circles with light teal fill and dark text labels. Level 2 children of VP: Adv node, V node each as circles with light orange fill and dark text, plus embedded NP node as circle with teal fill branching further into Det, Adj, N. Terminal nodes as leaves shown as rounded rectangles with light green fill and solid green border, text in serif font dark green, terminal words left to right "the" "clever" "cat" "quickly" "chased" "the" "frightened" "mouse". All connecting edges as straight lines in dark gray connecting parent to child with no arrow heads, lines should not cross ensuring proper tree layout. Color legend in bottom-left explaining blue for S, teal for NP, orange for VP, light colors for intermediate categories, green rectangles for terminal symbols. Derivation steps panel on right side showing the complete leftmost derivation sequence. Constituency brackets at bottom showing the bracketed representation. Clean academic illustration style, hierarchical tree layout with consistent spacing, balanced and symmetrical arrangement, professional typography throughout."""
    },
    
    "timeline": {
        "filename": "timeline.png",
        "prompt": """Create a stunning comprehensive horizontal timeline infographic depicting the complete history of formal language theory and the Chomsky hierarchy from 1930 to 2025. Ultra-wide format with subtle gradient background from warm cream on the left representing the past to cool light blue on the right representing the present, very light paper grain texture throughout. Main timeline spine as horizontal line at vertical center thick with gradient color from sepia brown on left through deep blue in middle to vibrant teal on right with subtle glow effect, small tick marks every decade with year labels below. Era sections as colored background bands: 1930-1945 "Foundations" warm sepia tint, 1945-1960 "Birth of Computer Science" light amber tint, 1960-1980 "Golden Age of Formal Languages" light blue tint, 1980-2000 "Applications and Extensions" light green tint, 2000-2025 "Modern Era" light purple tint. Milestone events alternating above and below the timeline. Above timeline theoretical developments: 1936 Turing Machine with small tape icon and Alan Turing silhouette in gold accent, 1943 McCulloch-Pitts Neurons with neural network icon in amber, 1956 Chomsky's "Three Models" as MAJOR milestone with three nested shapes icon and young Noam Chomsky silhouette in blue with larger visual treatment, 1959 Chomsky Hierarchy Formalized as MAJOR milestone largest visual treatment with four-level pyramid icon and small diagram of four types in blue, 1959 Rabin-Scott Theorem with DFA=NFA equation in indigo, 1961 Pumping Lemma with wavy line icon in violet, 1964 Kuroda's LBA Theorem with bounded tape icon in purple, 1985 Mildly Context-Sensitive with TAG tree icon in fuchsia, 2018-2023 Neural Networks and Formal Languages with neural network and formal language symbols icon in teal. Below timeline practical applications: 1948 Information Theory with entropy formula H icon and Claude Shannon silhouette in orange, 1957 Syntactic Structures as MAJOR milestone with book icon in red, 1959-1960 BNF and ALGOL with code syntax symbols in green, 1965 LR Parsing with parse table grid icon and Donald Knuth silhouette in emerald, 1967-1969 CYK Algorithm with triangular parsing table icon in teal, 1975 YACC with compiler icon in cyan, 1990s Model Checking with checkmark and automaton icon in sky blue, 2000s Bioinformatics with DNA helix and grammar symbols icon in blue. Key figures gallery as top banner showing circular portrait frames with silhouettes of Alan Turing, Claude Shannon, Noam Chomsky in largest central position, Michael Rabin and Dana Scott, Donald Knuth with names and years below each. Legend panel in bottom right with color key and symbol key and scale indicator. Title block in top left "The Evolution of Formal Language Theory" in elegant display font, subtitle "From Turing Machines to Neural Networks: A 90-Year Journey" with decorative underline. Elegant museum-quality infographic design, consistent iconography throughout, balanced composition, rich but not overwhelming color palette, all text highly legible, suitable for large-format poster printing."""
    },
    
    "comparison_chart": {
        "filename": "comparison_chart.png",
        "prompt": """Create a visually stunning comprehensive comparison infographic showing the complete correspondence between the four levels of the Chomsky Hierarchy including grammar types, automaton models, example languages, closure properties, and computational complexity. Clean white background and very subtle geometric hexagonal grid pattern in faint gray, elegant double-line border from edge. Title section at top with main title "The Chomsky Hierarchy: Complete Reference" in bold serif font, subtitle "Grammar Types, Automata, and Their Properties" in light weight, decorative line separator with small diamond ornament in center. Main content as four-row comparison table with seven columns: Type, Grammar, Production Rules, Automaton, Example Languages, Closure Properties, Complexity. Column headers in top row with dark charcoal background and white bold text. Row 1 Type-3 Regular with light red gradient background and solid red left accent bar, large "3" with "Regular" below in red, "Regular Grammar" cell, rule format "A to aB, A to a, A to epsilon" as mathematical expressions, "Finite Automaton DFA/NFA" with simple 3-state automaton icon and note "DFA = NFA", examples "(ab)*, a*b*, even number of a's, identifiers", closure properties all checkmarks in green for union intersection complement concatenation Kleene star reversal homomorphism, complexity "O(n) time O(1) space, Emptiness Decidable, Equivalence Decidable". Row 2 Type-2 Context-Free with light orange gradient background and solid orange accent bar, "Context-Free Grammar CFG", rule format "A to alpha", "Pushdown Automaton PDA" with stack icon and note "NPDA not equal DPDA", examples "{a^n b^n}, Dyck language, programming syntax, {ww^R}", closure checkmarks for union concatenation Kleene star but red X for intersection and complement, complexity "O(n^3) CYK, O(n) for DCFL, Emptiness Decidable, Equivalence Undecidable". Row 3 Type-1 Context-Sensitive with light teal gradient background and solid teal accent bar, "Context-Sensitive Grammar CSG", rule format "alpha A beta to alpha gamma beta where |gamma| >= 1 non-contracting", "Linear Bounded Automaton LBA" with bounded tape icon and note "NLBA =? DLBA open problem", examples "{a^n b^n c^n}, {ww}, {a^(n^2)}", closure all checkmarks, complexity "PSPACE-complete, Decidable but expensive, Emptiness Undecidable". Row 4 Type-0 Recursively Enumerable with light blue gradient background and solid blue accent bar, "Unrestricted Grammar UG", rule format "alpha to beta where alpha contains at least one non-terminal no restrictions", "Turing Machine TM" with infinite tape icon and note "Universal computation", examples "Halting problem language, any r.e. language, Universal TM language", closure checkmarks for union intersection concatenation Kleene star but red X for complement, complexity "Semi-decidable, May not halt on non-members, Most properties undecidable". Containment diagram on right side panel showing nested ovals Type-3 subset Type-2 subset Type-1 subset Type-0 color-coded to match rows with arrows indicating increasing power. Key relationships box in bottom left showing "Regular proper subset DCFL proper subset CFL proper subset CSL proper subset Recursive proper subset R.E." each class in corresponding color. Historical notes box in bottom right with hierarchy establishment dates and key contributors. Clean professional reference chart aesthetic, consistent typography sans-serif for headers serif for content, high information density but not cluttered, color-blind friendly palette, suitable for poster printing or digital reference."""
    }
}


# ============================================================
# DALL-E 3 Generator
# ============================================================

def generate_with_dalle3(prompts: dict, output_dir: Path):
    """Generate images using OpenAI DALL-E 3 API."""
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI library not installed. Run: pip install openai")
        return {}
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-openai-api-key-here':
        print("❌ OpenAI API key not configured in api_keys.txt")
        return {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)
    results = {}
    
    for i, (name, config) in enumerate(prompts.items(), 1):
        print(f"\n[{i}/{len(prompts)}] Generating: {name}")
        
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=config["prompt"],
                n=1,
                size="1792x1024",
                quality="hd"
            )
            
            image_url = response.data[0].url
            image_response = requests.get(image_url)
            image_path = output_dir / config["filename"]
            
            with open(image_path, "wb") as f:
                f.write(image_response.content)
            
            print(f"  ✓ Saved: {image_path.name}")
            results[name] = str(image_path)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[name] = None
    
    return results


# ============================================================
# Gemini Generator
# ============================================================

def generate_with_gemini(prompts: dict, output_dir: Path):
    """Generate images using Google Gemini/Imagen API."""
    if not GEMINI_AVAILABLE:
        print("❌ Google GenAI library not installed.")
        print("   For image generation, install: pip install google-genai")
        print("   (Note: google-generativeai package does NOT support image generation)")
        return {}
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'your-nanobananapro-api-key-here':
        print("❌ Gemini API key not configured in api_keys.txt")
        return {}
    
    print(f"   Using SDK: {GEMINI_SDK_TYPE}")
    
    # Check if we have the right SDK for image generation
    if GEMINI_SDK_TYPE == "google-generativeai":
        print("⚠️  Warning: google-generativeai package has limited image generation support.")
        print("   For full image generation capability, install: pip install google-genai")
        print("   Attempting to use Gemini 2.0 Flash for image generation...")
        return _generate_with_gemini_legacy(prompts, output_dir)
    
    # Use google-genai SDK (preferred for image generation)
    return _generate_with_gemini_new(prompts, output_dir)


def _generate_with_gemini_new(prompts: dict, output_dir: Path):
    """Generate images using google-genai SDK (newer, supports Imagen)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"❌ Failed to create Gemini client: {e}")
        return {}
    
    results = {}
    
    # Available models for image generation:
    # - imagen-3.0-generate-002 (Imagen 3)
    # - gemini-2.0-flash-exp (Gemini 2.0 with image output)
    model_name = "gemini-2.0-flash-exp"
    
    for i, (name, config) in enumerate(prompts.items(), 1):
        print(f"\n[{i}/{len(prompts)}] Generating: {name}")
        
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=config["prompt"],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                ),
            )
            
            # Extract image from response
            image_saved = False
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Save image from inline data
                    image_data = base64.b64decode(part.inline_data.data)
                    image_path = output_dir / config["filename"]
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    print(f"  ✓ Saved: {image_path.name}")
                    results[name] = str(image_path)
                    image_saved = True
                    break
            
            if not image_saved:
                print(f"  ✗ No image in response")
                results[name] = None
            
            # Small delay to avoid rate limiting
            if i < len(prompts):
                time.sleep(2)
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[name] = None
    
    return results


def _generate_with_gemini_legacy(prompts: dict, output_dir: Path):
    """Generate images using google-generativeai SDK (legacy, limited support)."""
    import google.generativeai as genai_legacy
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        genai_legacy.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"❌ Failed to configure Gemini: {e}")
        return {}
    
    results = {}
    
    # Try Gemini 2.0 Flash which has experimental image generation
    model_name = "gemini-2.0-flash-exp"
    
    try:
        model = genai_legacy.GenerativeModel(model_name)
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        print("   Try installing google-genai for better image generation support:")
        print("   pip install google-genai")
        return {}
    
    for i, (name, config) in enumerate(prompts.items(), 1):
        print(f"\n[{i}/{len(prompts)}] Generating: {name}")
        
        try:
            response = model.generate_content(
                config["prompt"],
                generation_config=genai_legacy.GenerationConfig(
                    response_mime_type="image/png"
                )
            )
            
            # Try to extract image from response
            image_saved = False
            if hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_data = base64.b64decode(part.inline_data.data)
                        image_path = output_dir / config["filename"]
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        print(f"  ✓ Saved: {image_path.name}")
                        results[name] = str(image_path)
                        image_saved = True
                        break
            
            if not image_saved:
                print(f"  ✗ No image in response (model may not support image generation)")
                results[name] = None
            
            # Small delay to avoid rate limiting
            if i < len(prompts):
                time.sleep(2)
                
        except Exception as e:
            error_msg = str(e)
            if "not supported" in error_msg.lower() or "invalid" in error_msg.lower():
                print(f"  ✗ Model does not support image generation")
                print(f"     Install google-genai for Imagen support: pip install google-genai")
            else:
                print(f"  ✗ Error: {e}")
            results[name] = None
    
    return results


# ============================================================
# Main
# ============================================================

def check_setup():
    """Check and display the current setup status."""
    print("\n" + "=" * 60)
    print("Setup Status Check")
    print("=" * 60)
    
    # Check OpenAI
    print("\n📦 OpenAI DALL-E 3:")
    if OPENAI_AVAILABLE:
        print("   ✓ openai library installed")
        if OPENAI_API_KEY and OPENAI_API_KEY != 'your-openai-api-key-here':
            print(f"   ✓ API key configured (sk-...{OPENAI_API_KEY[-8:]})")
        else:
            print("   ✗ API key NOT configured in api_keys.txt")
    else:
        print("   ✗ openai library NOT installed")
        print("   → Install with: pip install openai")
    
    # Check Gemini
    print("\n📦 Google Gemini/Imagen:")
    if GEMINI_AVAILABLE:
        print(f"   ✓ {GEMINI_SDK_TYPE} library installed")
        if GEMINI_SDK_TYPE == "google-generativeai":
            print("   ⚠️  This SDK has LIMITED image generation support")
            print("   → For full support, install: pip install google-genai")
        if GEMINI_API_KEY and GEMINI_API_KEY != 'your-nanobananapro-api-key-here':
            print(f"   ✓ API key configured (...{GEMINI_API_KEY[-8:]})")
        else:
            print("   ✗ API key NOT configured in api_keys.txt")
    else:
        print("   ✗ No Gemini library installed")
        print("   → Install with: pip install google-genai")
    
    # Check requests
    print("\n📦 Other dependencies:")
    try:
        import requests
        print("   ✓ requests library installed")
    except ImportError:
        print("   ✗ requests library NOT installed")
        print("   → Install with: pip install requests")
    
    print("\n" + "=" * 60)
    

def main():
    parser = argparse.ArgumentParser(
        description="Generate images for Chomsky Hierarchy paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_images.py --model dalle3    # Generate with DALL-E 3 only
    python generate_images.py --model gemini    # Generate with Gemini only
    python generate_images.py --model both      # Generate with both models
    python generate_images.py --list            # List available prompts
    python generate_images.py --check           # Check setup status
        """
    )
    parser.add_argument(
        '--model', 
        choices=['dalle3', 'gemini', 'both'],
        default='both',
        help='Which model to use for generation (default: both)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available image prompts'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check setup status (API keys, libraries)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Generate only a specific prompt (by name). Use --list to see available prompts.'
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_setup()
        return
    
    if args.list:
        print("\nAvailable image prompts:")
        print("=" * 40)
        for name in PROMPTS:
            print(f"  • {name}")
        return
    
    # Filter prompts if specific prompt is requested
    if args.prompt:
        if args.prompt not in PROMPTS:
            print(f"\n❌ Error: Prompt '{args.prompt}' not found.")
            print("\nAvailable prompts:")
            for name in PROMPTS:
                print(f"  • {name}")
            return
        prompts_to_generate = {args.prompt: PROMPTS[args.prompt]}
    else:
        prompts_to_generate = PROMPTS
    
    print("=" * 60)
    print("Image Generator for Brain Encoding Paper")
    print("=" * 60)
    print(f"\nImages to generate: {len(prompts_to_generate)}")
    if args.prompt:
        print(f"Selected prompt: {args.prompt}")
    print(f"Model(s): {args.model}")
    
    # Show SDK status
    print("\nSDK Status:")
    print(f"  • OpenAI: {'✓ Available' if OPENAI_AVAILABLE else '✗ Not installed'}")
    print(f"  • Gemini: {'✓ Available (' + GEMINI_SDK_TYPE + ')' if GEMINI_AVAILABLE else '✗ Not installed'}")
    
    if args.model in ['dalle3', 'both']:
        print("\n" + "=" * 60)
        print("Generating with DALL-E 3")
        print("=" * 60)
        print(f"Output: {DALLE3_DIR}")
        dalle3_results = generate_with_dalle3(prompts_to_generate, DALLE3_DIR)
        successful = sum(1 for v in dalle3_results.values() if v)
        print(f"\nDALL-E 3: {successful}/{len(prompts_to_generate)} images generated")
    
    if args.model in ['gemini', 'both']:
        print("\n" + "=" * 60)
        print("Generating with Gemini/Imagen")
        print("=" * 60)
        print(f"Output: {GEMINI_DIR}")
        gemini_results = generate_with_gemini(prompts_to_generate, GEMINI_DIR)
        successful = sum(1 for v in gemini_results.values() if v)
        print(f"\nGemini: {successful}/{len(prompts_to_generate)} images generated")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nImages saved to:")
    if args.model in ['dalle3', 'both']:
        print(f"  • DALL-E 3: {DALLE3_DIR}/")
    if args.model in ['gemini', 'both']:
        print(f"  • Gemini:   {GEMINI_DIR}/")


if __name__ == "__main__":
    main()
