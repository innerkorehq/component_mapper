from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRoutingConfig(BaseModel):
    complex_model: str = "anthropic/claude-sonnet-4-6"
    standard_model: str = "anthropic/claude-haiku-4-5-20251001"
    fast_model: str = "anthropic/claude-haiku-4-5-20251001"

    complex_candidate_threshold: int = 3
    complex_prop_count_threshold: int = 8
    fast_max_candidates: int = 1
    fast_max_props: int = 4


class SignatureIndexConfig(BaseModel):
    index_cache_path: str = ".cache/signature_index.json"
    custom_registry_path: str = ".cache/custom_registry.json"
    tfidf_max_features: int = 512
    tfidf_ngram_range: tuple[int, int] = (2, 4)

    direct_match_threshold: float = 0.85
    candidate_min_threshold: float = 0.40
    max_candidates_per_segment: int = 4


class MappingCacheConfig(BaseModel):
    cache_path: str = ".cache/mapping_cache.json"
    auto_persist_every: int = 50
    min_confidence_to_cache: float = 0.0


class MCPConfig(BaseModel):
    transport: str = "stdio"
    sse_url: str = "http://localhost:7423/sse"
    components_json_path: str = "components.json"
    startup_timeout_seconds: int = 30


class ExternalRegistryConfig(BaseModel):
    """
    An open-source shadcn-compatible registry from https://ui.shadcn.com/docs/directory.

    name         — short identifier used as a component name prefix (e.g. "bundui")
    namespace    — shadcn CLI namespace used to install components (e.g. "@bundui")
    url_template — URL with {name} placeholder (e.g. "https://bundui.io/r/{name}.json")
    components   — component names to pre-fetch during index rebuild; empty = registry
                   is registered for on-demand use but not pre-indexed at startup
    description  — human-readable description from the directory listing
    open_source  — always True for entries in this list; field kept for future filtering
    """
    name: str
    namespace: str
    url_template: str
    components: list[str] = Field(default_factory=list)
    description: str = ""
    open_source: bool = True

    @computed_field
    @property
    def registry_index_url(self) -> str:
        """URL to this registry's index (registry.json).
        Derived by replacing {name} with 'registry' in url_template."""
        return self.url_template.replace("{name}", "registry")


# ---------------------------------------------------------------------------
# All open-source registries from https://ui.shadcn.com/r/registries.json
# Source: ui.shadcn.com/docs/directory — all entries are open source by policy.
#
# components field is populated only for registries where specific components
# have been verified to exist (avoids 404 storms during index rebuild).
# Add component names here as you confirm they are available in that registry.
# ---------------------------------------------------------------------------
OPEN_SOURCE_REGISTRIES: list[ExternalRegistryConfig] = [
    ExternalRegistryConfig(name="8bitcn",          namespace="@8bitcn",          url_template="https://www.8bitcn.com/r/{name}.json",                              description="8-bit styled retro components. Works with favorite frameworks. Open Source."),
    ExternalRegistryConfig(name="8starlabs-ui",    namespace="@8starlabs-ui",    url_template="https://ui.8starlabs.com/r/{name}.json",                           description="Beautifully designed components with niche, high-utility UI elements."),
    ExternalRegistryConfig(name="unlumen-ui",       namespace="@unlumen-ui",       url_template="https://ui.unlumen.com/r/{name}.json",                              description="Primitives and components with animation and design focus."),
    ExternalRegistryConfig(name="abui",             namespace="@abui",             url_template="https://abui.io/r/{name}.json",                                     description="Shadcn-compatible registry of reusable components and utilities."),
    ExternalRegistryConfig(name="arc",              namespace="@arc",              url_template="https://witharc.co/r/{name}.json",                                  description="Animated, accessible UI built with React and Tailwind CSS."),
    ExternalRegistryConfig(name="aceternity",       namespace="@aceternity",       url_template="https://ui.aceternity.com/registry/{name}.json",                    description="Modern component library with Tailwind CSS and Motion for React."),
    ExternalRegistryConfig(name="aevr",             namespace="@aevr",             url_template="https://ui.aevr.space/r/{name}.json",                               description="Focused, production-ready components for React/Next.js projects."),
    ExternalRegistryConfig(name="ai-blocks",        namespace="@ai-blocks",        url_template="https://webllm.org/r/{name}.json",                                  description="AI components for web without server or API keys."),
    ExternalRegistryConfig(name="ai-elements",      namespace="@ai-elements",      url_template="https://ai-sdk.dev/elements/api/registry/{name}.json",              description="Pre-built components for building AI-native applications faster."),
    ExternalRegistryConfig(name="aliimam",          namespace="@aliimam",          url_template="https://aliimam.in/r/{name}.json",                                  description="Digital experiences connecting and inspiring end-to-end development."),
    ExternalRegistryConfig(name="amplo",            namespace="@amplo",            url_template="https://amplo.ale.design/r/{name}.json",                            description="OKLCH-native fill picker with WCAG contrast metrics and accessibility."),
    ExternalRegistryConfig(name="animate-ui",       namespace="@animate-ui",       url_template="https://animate-ui.com/r/{name}.json",                              description="Fully animated, open-source React component distribution library."),
    ExternalRegistryConfig(name="assistant-ui",     namespace="@assistant-ui",     url_template="https://r.assistant-ui.com/{name}.json",                            description="Radix-style React primitives for AI chat with multiple backend adapters."),
    ExternalRegistryConfig(name="tool-ui",          namespace="@tool-ui",          url_template="https://www.tool-ui.com/r/{name}.json",                             description="Open source React components for AI tool calls and assistant outputs."),
    ExternalRegistryConfig(name="better-upload",    namespace="@better-upload",    url_template="https://better-upload.com/r/{name}.json",                           description="Simple file uploads for React with S3-compatible service support."),
    ExternalRegistryConfig(name="basecn",           namespace="@basecn",           url_template="https://basecn.dev/r/{name}.json",                                  description="Beautifully crafted shadcn/ui components powered by Base UI."),
    ExternalRegistryConfig(name="billingsdk",       namespace="@billingsdk",       url_template="https://billingsdk.com/r/{name}.json",                              description="Open-source React and Next.js components for SaaS billing and payments."),
    ExternalRegistryConfig(name="blocks-so",        namespace="@blocks-so",        url_template="https://blocks.so/r/{name}.json",                                   description="Clean, modern application building blocks. Free and Open Source."),
    ExternalRegistryConfig(name="boldkit",          namespace="@boldkit",          url_template="https://boldkit.dev/r/{name}.json",                                  description="Neubrutalism component library with thick borders and hard shadows."),
    ExternalRegistryConfig(name="bundui",           namespace="@bundui",           url_template="https://bundui.io/r/{name}.json",                                   description="150+ handcrafted UI components built with Tailwind CSS and shadcn/ui.", components=["pagination"]),
    ExternalRegistryConfig(name="cardcn",           namespace="@cardcn",           url_template="https://cardcn.dev/r/{name}.json",                                  description="Beautifully-designed shadcn card components collection."),
    ExternalRegistryConfig(name="chamaac",          namespace="@chamaac",          url_template="https://chamaac.com/r/{name}.json",                                 description="Beautiful, animated components to elevate web projects instantly."),
    ExternalRegistryConfig(name="clerk",            namespace="@clerk",            url_template="https://clerk.com/r/{name}.json",                                   description="Easiest way to add authentication and user management to applications."),
    ExternalRegistryConfig(name="cognicatch",       namespace="@cognicatch",       url_template="https://cognicatch.dev/registry/{name}.json",                       description="Adaptive Error Boundaries and graceful fallback UIs."),
    ExternalRegistryConfig(name="commercn",         namespace="@commercn",         url_template="https://commercn.com/r/{name}.json",                                description="Shadcn UI Blocks specifically for Ecommerce websites."),
    ExternalRegistryConfig(name="coss",             namespace="@coss",             url_template="https://coss.com/ui/r/{name}.json",                                 description="Modern UI component library built on Base UI for developers and AI."),
    ExternalRegistryConfig(name="creative-tim",     namespace="@creative-tim",     url_template="https://www.creative-tim.com/ui/r/{name}.json",                     description="Open-source UI components, blocks and AI Agents for integration."),
    ExternalRegistryConfig(name="cult-ui",          namespace="@cult-ui",          url_template="https://cult-ui.com/r/{name}.json",                                 description="Rare, curated shadcn-compatible components with Framer Motion."),
    ExternalRegistryConfig(name="diceui",           namespace="@diceui",           url_template="https://diceui.com/r/{name}.json",                                  description="Accessible shadcn/ui components built with React and Tailwind CSS."),
    ExternalRegistryConfig(name="doras-ui",         namespace="@doras-ui",         url_template="https://ui.doras.to/r/{name}.json",                                 description="Beautiful, reusable component blocks built with React."),
    ExternalRegistryConfig(name="dsikeres1",        namespace="@dsikeres1",        url_template="https://dsikeres1.github.io/react-date-range-picker/r/{name}.json", description="Headless, composable date and date range picker with zero dependencies."),
    ExternalRegistryConfig(name="elements",         namespace="@elements",         url_template="https://www.tryelements.dev/r/{name}.json",                         description="Full-stack shadcn/ui components with auth, monetization, and AI."),
    ExternalRegistryConfig(name="elevenlabs-ui",    namespace="@elevenlabs-ui",    url_template="https://ui.elevenlabs.io/r/{name}.json",                            description="Open Source agent and audio components for customization."),
    ExternalRegistryConfig(name="efferd",           namespace="@efferd",           url_template="https://efferd.com/r/{name}.json",                                  description="Beautifully crafted Shadcn/UI blocks for modern websites."),
    ExternalRegistryConfig(name="einui",            namespace="@einui",            url_template="https://ui.eindev.ir/r/{name}.json",                                description="Beautiful, responsive Shadcn components with frosted glass morphism."),
    ExternalRegistryConfig(name="eldoraui",         namespace="@eldoraui",         url_template="https://eldoraui.site/r/{name}.json",                               description="Modern UI component library with TypeScript and Framer Motion."),
    ExternalRegistryConfig(name="evilcharts",       namespace="@evilcharts",       url_template="https://evilcharts.com/r/{name}.json",                              description="Open-source chart UI website with shadcn and Recharts."),
    ExternalRegistryConfig(name="formcn",           namespace="@formcn",           url_template="https://formcn.dev/r/{name}.json",                                  description="Build production-ready forms using shadcn components and modern tools."),
    ExternalRegistryConfig(name="gaia",             namespace="@gaia",             url_template="https://ui.heygaia.io/r/{name}.json",                               description="Production-ready UI for building beautiful AI assistants."),
    ExternalRegistryConfig(name="gamifykit",        namespace="@gamifykit",        url_template="https://gamifykit.com/r/{name}.json",                               description="Fully composable components extending shadcn/ui with gamification patterns."),
    ExternalRegistryConfig(name="glass-ui",         namespace="@glass-ui",         url_template="https://glass-ui.crenspire.com/r/{name}.json",                      description="40+ glassmorphic React components with Apple-inspired design."),
    ExternalRegistryConfig(name="glasscn",          namespace="@glasscn",          url_template="https://glasscn-components.vercel.app/r/{name}.json",               description="Shadcn-compatible glassmorphism components inspired by Apple."),
    ExternalRegistryConfig(name="hextaui",          namespace="@hextaui",          url_template="https://hextaui.com/r/{name}.json",                                 description="Ready-to-use foundation components built on shadcn/ui.", components=["pagination"]),
    ExternalRegistryConfig(name="shadcnhooks",      namespace="@shadcnhooks",      url_template="https://shadcn-hooks.com/r/{name}.json",                            description="Comprehensive React Hooks Collection built with Shadcn."),
    ExternalRegistryConfig(name="indiacn",          namespace="@indiacn",          url_template="https://indiacn.in/r/{name}.json",                                  description="UX4G 2.0 design system for India with accessible components."),
    ExternalRegistryConfig(name="intentui",         namespace="@intentui",         url_template="https://intentui.com/r/{name}",                                     description="Accessible React component library to copy and customize."),
    ExternalRegistryConfig(name="kibo-ui",          namespace="@kibo-ui",          url_template="https://www.kibo-ui.com/r/{name}.json",                             description="Composable, accessible, open source components for shadcn/ui."),
    ExternalRegistryConfig(name="kanpeki",          namespace="@kanpeki",          url_template="https://kanpeki.vercel.app/r/{name}.json",                          description="Perfect-designed components built on React Aria and Motion."),
    ExternalRegistryConfig(name="kokonutui",        namespace="@kokonutui",        url_template="https://kokonutui.com/r/{name}.json",                               description="Stunning components with Tailwind CSS, shadcn/ui and Motion."),
    ExternalRegistryConfig(name="launchui",         namespace="@launchui",         url_template="https://www.launchuicomponents.com/r/{name}.json",                  description="Landing page components and templates with React and Shadcn/ui."),
    ExternalRegistryConfig(name="limeplay",         namespace="@limeplay",         url_template="https://limeplay.winoffrg.dev/r/{name}.json",                       description="Modern UI Library for building media players in React."),
    ExternalRegistryConfig(name="loading-ui",       namespace="@loading-ui",       url_template="https://loading-ui.com/r/{name}.json",                              description="Spinners, loaders, and animations for modern web apps."),
    ExternalRegistryConfig(name="lucide-animated",  namespace="@lucide-animated",  url_template="https://lucide-animated.com/r/{name}.json",                         description="Open-source smooth animated lucide icons collection."),
    ExternalRegistryConfig(name="lytenyte",         namespace="@lytenyte",         url_template="https://www.1771technologies.com/r/{name}.json",                     description="High performance, lightweight React data grid with Shadcn theming."),
    ExternalRegistryConfig(name="magicui",          namespace="@magicui",          url_template="https://magicui.design/r/{name}",                                   description="150+ free animated components and effects for design engineers."),
    ExternalRegistryConfig(name="manifest",         namespace="@manifest",         url_template="https://ui.manifest.build/r/{name}.json",                           description="Agentic UI toolkit for building MCP Apps with open-source components."),
    ExternalRegistryConfig(name="mapcn",            namespace="@mapcn",            url_template="https://mapcn.dev/r/{name}.json",                                   description="Customizable map components for React built on MapLibre."),
    ExternalRegistryConfig(name="moleculeui",       namespace="@moleculeui",       url_template="https://www.moleculeui.design/r/{name}.json",                       description="Modern React component library focused on intuitive interactions."),
    ExternalRegistryConfig(name="motion-primitives",namespace="@motion-primitives",url_template="https://motion-primitives.com/c/{name}.json",                      description="Beautifully designed motions components for copy-paste use."),
    ExternalRegistryConfig(name="nordaun",          namespace="@nordaun",          url_template="https://ui.nordaun.com/r/{name}.json",                              description="Simple components for extraordinary creations."),
    ExternalRegistryConfig(name="ncdai",            namespace="@ncdai",            url_template="https://chanhdai.com/r/{name}.json",                                description="Pixel-perfect, uniquely crafted components collection."),
    ExternalRegistryConfig(name="nuqs",             namespace="@nuqs",             url_template="https://nuqs.dev/r/{name}.json",                                    description="Custom parsers and utilities for type-safe URL state management."),
    ExternalRegistryConfig(name="neobrutalism",     namespace="@neobrutalism",     url_template="https://www.neobrutalism.dev/r/{name}.json",                        description="Neobrutalism-styled components based on shadcn/ui."),
    ExternalRegistryConfig(name="nessra-ui",        namespace="@nessra-ui",        url_template="https://nessra-ui.vercel.app/r/{name}.json",                        description="Beautiful, accessible components with Tailwind CSS v4 and Radix."),
    ExternalRegistryConfig(name="openstatus",       namespace="@openstatus",       url_template="https://openstatus.dev/r/{name}.json",                              description="Hand-crafted, accessible components for beautiful status pages."),
    ExternalRegistryConfig(name="optics",           namespace="@optics",           url_template="https://optics.agusmayol.com.ar/r/{name}.json",                     description="Design system with re-styled components, utilities, and hooks."),
    ExternalRegistryConfig(name="oui",              namespace="@oui",              url_template="https://oui.mw10013.workers.dev/r/{name}.json",                     description="React Aria Components with shadcn characteristics."),
    ExternalRegistryConfig(name="pacekit",          namespace="@pacekit",          url_template="https://ui.pacekit.dev/r/{name}.json",                              description="UI blocks for real apps and dashboards from early ideas to production."),
    ExternalRegistryConfig(name="pacekit-gsap",     namespace="@pacekit-gsap",     url_template="https://gsap.pacekit.dev/r/{name}.json",                            description="Animated GSAP components for smooth interaction and rich detail."),
    ExternalRegistryConfig(name="plate",            namespace="@plate",            url_template="https://platejs.org/r/{name}.json",                                 description="AI-powered rich text editor for React."),
    ExternalRegistryConfig(name="prompt-kit",       namespace="@prompt-kit",       url_template="https://www.prompt-kit.com/c/{name}.json",                          description="Core building blocks for AI apps with accessible components."),
    ExternalRegistryConfig(name="prosekit",         namespace="@prosekit",         url_template="https://prosekit.dev/r/{name}.json",                                description="Powerful flexible rich text editor for multiple frameworks."),
    ExternalRegistryConfig(name="react-aria",       namespace="@react-aria",       url_template="https://react-aria.adobe.com/registry/{name}.json",                 description="Customizable components with adaptive interactions and accessibility."),
    ExternalRegistryConfig(name="react-bits",       namespace="@react-bits",       url_template="https://reactbits.dev/r/{name}.json",                               description="Animated, interactive, customizable React components for websites."),
    ExternalRegistryConfig(name="retroui",          namespace="@retroui",          url_template="https://retroui.dev/r/{name}.json",                                 description="Neobrutalism styled React and TailwindCSS library for bold apps."),
    ExternalRegistryConfig(name="reui",             namespace="@reui",             url_template="https://reui.io/r/{name}.json",                                     description="Free library of 1,000+ components and patterns for shadcn."),
    ExternalRegistryConfig(name="scrollxui",        namespace="@scrollxui",        url_template="https://www.scrollxui.dev/registry/{name}.json",                    description="Animated, interactive, customizable component library for ShadCN."),
    ExternalRegistryConfig(name="spell",            namespace="@spell",            url_template="https://spell.sh/r/{name}.json",                                    description="Beautiful, sophisticated UI components for modern React applications."),
    ExternalRegistryConfig(name="square-ui",        namespace="@square-ui",        url_template="https://square.lndev.me/registry/{name}.json",                      description="Beautifully crafted open-source layouts built with shadcn/ui."),
    ExternalRegistryConfig(name="roiui",            namespace="@roiui",            url_template="https://roiui.com/r/{name}.json",                                   description="UI components and blocks built with Base UI primitives."),
    ExternalRegistryConfig(name="satoriui",         namespace="@satoriui",         url_template="https://satoriui.site/r/{name}.json",                               description="Comprehensive high-fidelity interaction components with motion."),
    ExternalRegistryConfig(name="solaceui",         namespace="@solaceui",         url_template="https://www.solaceui.com/r/{name}.json",                            description="Production-ready sections and templates for Next.js and Motion."),
    ExternalRegistryConfig(name="shadcnblocks",     namespace="@shadcnblocks",     url_template="https://shadcnblocks.com/r/{name}.json",                            description="1429 blocks, 1189 variants, 14 templates, themes and admin patterns."),
    ExternalRegistryConfig(name="shadcndesign",     namespace="@shadcndesign",     url_template="https://shadcndesign-free.vercel.app/r/{name}.json",                description="Growing collection of high-quality blocks and themes for shadcn/ui."),
    ExternalRegistryConfig(name="shadcnmaps",       namespace="@shadcnmaps",       url_template="https://shadcnmaps.com/r/{name}.json",                              description="Beautiful map components powered by pure SVG."),
    ExternalRegistryConfig(name="shadcnstore",      namespace="@shadcnstore",      url_template="https://shadcnstore.com/r/{name}.json",                             description="Growing collection of shadcn/ui components, blocks, and templates."),
    ExternalRegistryConfig(name="shadcn-studio",    namespace="@shadcn-studio",    url_template="https://shadcnstudio.com/r/{name}.json",                            description="Open-source shadcn/ui components with powerful theme generator."),
    ExternalRegistryConfig(name="shadcn-editor",    namespace="@shadcn-editor",    url_template="https://raw.githubusercontent.com/htmujahid/shadcn-editor/refs/heads/main/public/r/{name}.json", description="Accessible, customizable rich text editor with Lexical and Shadcn."),
    ExternalRegistryConfig(name="shadcnuikit",      namespace="@shadcnuikit",      url_template="https://shadcnuikit.com/r/{name}.json",                             description="Admin dashboards, website templates, and real-world examples."),
    ExternalRegistryConfig(name="shadcncraft",      namespace="@shadcncraft",      url_template="https://shadcncraft.com/r/{name}.json",                             description="Polished shadcn/ui components built to production standards."),
    ExternalRegistryConfig(name="shark",            namespace="@shark",            url_template="https://shark.vini.one/r/{name}.json",                              description="Shadcn/ui-style components built on Ark UI."),
    ExternalRegistryConfig(name="smoothui",         namespace="@smoothui",         url_template="https://smoothui.dev/r/{name}.json",                                description="Motion components with Framer Motion and TailwindCSS animations."),
    ExternalRegistryConfig(name="spectrumui",       namespace="@spectrumui",       url_template="https://ui.spectrumhq.in/r/{name}.json",                            description="Modern component library with elegant, responsive components."),
    ExternalRegistryConfig(name="supabase",         namespace="@supabase",         url_template="https://supabase.com/ui/r/{name}.json",                             description="React components and blocks connecting front-end to Supabase back-end."),
    ExternalRegistryConfig(name="tailark",          namespace="@tailark",          url_template="https://tailark.com/r/{name}.json",                                 description="Shadcn blocks designed for building modern marketing websites."),
    ExternalRegistryConfig(name="taki",             namespace="@taki",             url_template="https://taki-ui.com/r/{name}.json",                                 description="Accessible components built with React Aria and Shadcn tokens."),
    ExternalRegistryConfig(name="thegridcn",        namespace="@thegridcn",        url_template="https://thegridcn.com/r/{name}.json",                               description="Tron-inspired shadcn/ui theme with sci-fi components."),
    ExternalRegistryConfig(name="uitripled",        namespace="@uitripled",        url_template="https://ui.tripled.work/r/{name}.json",                             description="Production-ready UI components and blocks with Framer Motion."),
    ExternalRegistryConfig(name="utilcn",           namespace="@utilcn",           url_template="https://utilcn.dev/r/{name}.json",                                  description="Fullstack registry items for big features and ChatGPT apps."),
    ExternalRegistryConfig(name="pureui",           namespace="@pureui",           url_template="https://pure.kam-ui.com/r/{name}.json",                             description="Refined, animated, accessible components with Base UI and Motion."),
    ExternalRegistryConfig(name="tailwind-builder", namespace="@tailwind-builder", url_template="https://tailwindbuilder.ai/r/{name}.json",                          description="Free UI blocks and AI tools for forms, tables, and charts."),
    ExternalRegistryConfig(name="tailwind-admin",   namespace="@tailwind-admin",   url_template="https://tailwind-admin.com/r/{name}.json",                          description="Free tailwind admin dashboard templates and UI-blocks."),
    ExternalRegistryConfig(name="forgeui",          namespace="@forgeui",          url_template="https://forgeui.in/r/{name}.json",                                  description="Beautifully designed, accessible, customizable open-source components."),
    ExternalRegistryConfig(name="skiper-ui",        namespace="@skiper-ui",        url_template="https://skiper-ui.com/registry/{name}.json",                        description="Uncommon components for Next.js with shadcn CLI 3.0."),
    ExternalRegistryConfig(name="animbits",         namespace="@animbits",         url_template="https://animbits.dev/r/{name}.json",                                description="Animated UI components using Framer Motion with general-purpose effects."),
    ExternalRegistryConfig(name="shadcn-space",     namespace="@shadcn-space",     url_template="https://shadcnspace.com/r/{name}.json",                             description="Extra-ordinary, customizable shadcn/ui components and themes."),
    ExternalRegistryConfig(name="icons-animated",   namespace="@icons-animated",   url_template="https://icons.lndev.me/r/{name}.json",                              description="Open-source animated icons (Tabler, Phosphor) for projects."),
    ExternalRegistryConfig(name="heroicons-animated",namespace="@heroicons-animated",url_template="https://www.heroicons-animated.com/r/{name}.json",                description="316 beautifully animated heroicons for projects."),
    ExternalRegistryConfig(name="devl",             namespace="@devl",             url_template="https://devl.dev/r/{name}.json",                                    description="Hand-crafted layouts and UI primitives for shipping fast."),
    ExternalRegistryConfig(name="beste-ui",         namespace="@beste-ui",         url_template="https://ui.beste.co/r/{name}.json",                                 description="Production-ready UI blocks for landing pages and dashboards."),
    ExternalRegistryConfig(name="tokenui",          namespace="@tokenui",          url_template="https://www.tokenui.dev/r/{name}.json",                             description="Beautiful, interactive documentation components for design tokens."),
    ExternalRegistryConfig(name="lumiui",           namespace="@lumiui",           url_template="https://www.lumiui.dev/r/{name}.json",                              description="Composable React components with Base UI and Tailwind CSS."),
    ExternalRegistryConfig(name="uselayouts",       namespace="@uselayouts",       url_template="https://uselayouts.com/r/{name}.json",                              description="Premium animated React components and micro-interactions."),
    ExternalRegistryConfig(name="joyco",            namespace="@joyco",            url_template="https://registry.joyco.studio/r/{name}.json",                       description="Components including MobileMenu, ChatUI, and HLSVideoPlayer."),
    ExternalRegistryConfig(name="gooseui",          namespace="@gooseui",          url_template="https://gooseui.pro/r/{name}.json",                                 description="Open source with animated components and custom notifications."),
    ExternalRegistryConfig(name="baselayer",        namespace="@baselayer",        url_template="https://www.baselayer.dev/r/{name}.json",                           description="Components built on React Aria and Tailwind CSS."),
    ExternalRegistryConfig(name="jolyui",           namespace="@jolyui",           url_template="https://www.jolyui.dev/r/{name}.json",                              description="Modern React component library with TypeScript and Tailwind."),
    ExternalRegistryConfig(name="fab-ui",           namespace="@fab-ui",           url_template="https://fab-ui.com/r/{name}.json",                                  description="Beautifully designed UI components for modern web applications."),
    ExternalRegistryConfig(name="asanshay",         namespace="@asanshay",         url_template="https://ds.asanshay.com/r/{name}.json",                             description="Clean, beautiful, simple UI primitives and AI elements."),
    ExternalRegistryConfig(name="typedora-ui",      namespace="@typedora-ui",      url_template="https://typedora-ui.netlify.app/r/{name}.json",                     description="Next-generation extension layer for shadcn/ui with type-safety."),
    ExternalRegistryConfig(name="sona-ui",          namespace="@sona-ui",          url_template="https://sona-ui.vercel.app/r/{name}.json",                          description="Modern UI library with React and TailwindCSS for web applications."),
    ExternalRegistryConfig(name="pixelact-ui",      namespace="@pixelact-ui",      url_template="https://pixelactui.com/r/{name}.json",                              description="Playful pixel art style components built on shadcn."),
    ExternalRegistryConfig(name="emerald-ui",       namespace="@emerald-ui",       url_template="https://emerald-ui.com/r/{name}.json",                              description="Components with Motion, GSAP, Tailwind CSS and shadcn/ui."),
    ExternalRegistryConfig(name="componentry",      namespace="@componentry",      url_template="https://componentry.fun/r/{name}.json",                             description="Beautiful, interactive React and Tailwind components for UIs."),
    ExternalRegistryConfig(name="fluid",            namespace="@fluid",            url_template="https://www.fluidfunctionalism.com/r/{name}.json",                  description="Fluid components for functional clarity with spring animations."),
    ExternalRegistryConfig(name="gammaui",          namespace="@gammaui",          url_template="https://www.gammaui.com/r/{name}.json",                             description="Landing page components with React, Tailwind CSS and Motion."),
    ExternalRegistryConfig(name="tailgrids",        namespace="@tailgrids",        url_template="https://tailgrids.com/docs/r/{name}.json",                          description="React UI Components powered by Tailwind CSS."),
    ExternalRegistryConfig(name="nexus-ui",         namespace="@nexus-ui",         url_template="https://nexus-ui.dev/r/{name}.json",                                description="Open-source primitives for building AI interfaces with chat."),
    ExternalRegistryConfig(name="sabraman",         namespace="@sabraman",         url_template="https://sabraman.ru/r/{name}.json",                                 description="Legacy skeuomorphic UI components and blocks for shadcn."),
    ExternalRegistryConfig(name="odysseyui",        namespace="@odysseyui",        url_template="https://www.odysseyui.com/r/{name}.json",                           description="Design-focused component library for Next.js built for speed."),
    ExternalRegistryConfig(name="openpolicy",       namespace="@openpolicy",       url_template="https://www.openpolicy.sh/r/{name}.json",                           description="Open-source components for terms, privacy and cookie banners."),
    ExternalRegistryConfig(name="mksingh",          namespace="@mksingh",          url_template="https://mksingh.dev/r/{name}.json",                                 description="Personal registry of production-ready ShadCN components."),
    ExternalRegistryConfig(name="flowkit-ui",       namespace="@flowkit-ui",       url_template="https://flowkit-ui.vzkiss.com/r/{name}.json",                       description="Opinionated, accessible components on Base UI."),
    ExternalRegistryConfig(name="termcn",           namespace="@termcn",           url_template="https://termcn.vercel.app/r/{name}.json",                           description="Beautiful terminal UIs made simple. Customizable React components."),
    ExternalRegistryConfig(name="remocn",           namespace="@remocn",           url_template="https://www.remocn.dev/r/{name}.json",                              description="Production-ready components for Remotion with text animations."),
    ExternalRegistryConfig(name="aicanvas",         namespace="@aicanvas",         url_template="https://aicanvas.me/r/{name}.json",                                 description="54 animated React components with AI reproduction prompts."),
    ExternalRegistryConfig(name="delta",            namespace="@delta",            url_template="https://deltacomponents.dev/r/{name}.json",                         description="Shadcn registry for AI interfaces with chat and media."),
    ExternalRegistryConfig(name="evilbuttons",      namespace="@evilbuttons",      url_template="https://evilbuttons.radiumcoders.com/r/{name}.json",                description="Animated button collection built with Motion for interactive feedback."),
    ExternalRegistryConfig(name="stepper",          namespace="@stepper",          url_template="https://francozeta-stepper.vercel.app/{name}.json",                 description="Modern, accessible, composable Stepper for React."),
    ExternalRegistryConfig(name="framecn",          namespace="@framecn",          url_template="https://framecn.vercel.app/r/{name}.json",                          description="Beautiful videos made simple. Customizable video React components."),
    ExternalRegistryConfig(name="ui-layouts",       namespace="@ui-layouts",       url_template="https://ui-layouts.com/r/{name}.json",                              description="Components, effects, tools and blocks for modern interfaces."),
    ExternalRegistryConfig(name="uicapsule",        namespace="@uicapsule",        url_template="https://uicapsule.com/r/{name}.json",                               description="Curated components at intersection of AI/UI and design experiments."),
]


class RegistryConfig(BaseModel):
    shadcn_registry_base_url: str = "https://ui.shadcn.com/r/styles/new-york-v4"
    custom_registry_base_url: str = ""
    fetch_timeout_seconds: int = 10
    max_concurrent_fetches: int = 10
    http_cache_ttl_hours: int = 24
    # All open-source registries from the Shadcn community directory.
    # Registries with an empty components list are registered for on-demand use
    # but not pre-fetched during index rebuild (avoids unnecessary network requests).
    external_registries: list[ExternalRegistryConfig] = Field(
        default_factory=lambda: OPEN_SOURCE_REGISTRIES
    )


class LiteLLMConfig(BaseModel):
    batch_size: int = 15
    max_concurrent_batches: int = 4
    timeout_seconds: int = 60
    config_path: str = ""  # path to litellm_config.json; empty = no file
    api_key_env: str = "LLM_API_KEY"  # env var name for the universal API key


class MapperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MAPPER_")

    mcp: MCPConfig = MCPConfig()
    registry: RegistryConfig = RegistryConfig()
    signature_index: SignatureIndexConfig = SignatureIndexConfig()
    mapping_cache: MappingCacheConfig = MappingCacheConfig()
    model_routing: ModelRoutingConfig = ModelRoutingConfig()
    litellm: LiteLLMConfig = LiteLLMConfig()

    astro_project_root: str = "./output/astro"
    generate_collection_schemas: bool = True
