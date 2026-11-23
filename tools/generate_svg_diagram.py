import os

def generate_svg():
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 660" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <!-- Shadow Filter -->
    <filter id="card-shadow" x="-5%" y="-10%" width="110%" height="130%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.06"/>
    </filter>
    <filter id="pill-shadow" x="-10%" y="-20%" width="120%" height="150%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#3b0764" flood-opacity="0.25"/>
    </filter>
    <filter id="pending-shadow" x="-10%" y="-20%" width="120%" height="150%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#ea580c" flood-opacity="0.28"/>
    </filter>

    <!-- Arrow Markers -->
    <marker id="purple-arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <polygon points="1,1 7,4 1,7" fill="#4c1d95" />
    </marker>
    <marker id="orange-arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <polygon points="1,1 7,4 1,7" fill="#ea580c" />
    </marker>
  </defs>

  <!-- ==================== LEFT BRACKET (COMPLETED) ==================== -->
  <!-- Smooth curly bracket grouping cards 1 to 5 -->
  <path d="M 285 58 
           Q 245 58 245 100 
           L 245 200 
           Q 245 235 210 235 
           Q 245 235 245 270 
           L 245 370 
           Q 245 412 285 412" 
        fill="none" 
        stroke="#4c1d95" 
        stroke-width="4.5" 
        stroke-linecap="round" 
        stroke-linejoin="round" />

  <!-- "Completed" Badge -->
  <g filter="url(#pill-shadow)">
    <rect x="75" y="210" width="138" height="50" rx="25" fill="#3b0764" stroke="#4c1d95" stroke-width="2"/>
    <text x="144" y="242" fill="#ffffff" font-size="19" font-weight="800" text-anchor="middle" letter-spacing="0.5">Completed</text>
  </g>

  <!-- ==================== PENDING BADGE & ARROW ==================== -->
  <g filter="url(#pending-shadow)">
    <rect x="85" y="525" width="128" height="50" rx="25" fill="#ea580c" stroke="#c2410c" stroke-width="2"/>
    <text x="149" y="557" fill="#ffffff" font-size="19" font-weight="800" text-anchor="middle" letter-spacing="0.5">Pending</text>
  </g>

  <!-- Arrow pointing to Step 6 -->
  <line x1="223" y1="550" x2="285" y2="550" stroke="#ea580c" stroke-width="3.5" marker-end="url(#orange-arrow)" />


  <!-- ==================== DOWNWARD CONNECTING ARROWS ==================== -->
  <line x1="600" y1="95"  x2="600" y2="117" stroke="#4c1d95" stroke-width="3" marker-end="url(#purple-arrow)"/>
  <line x1="600" y1="185" x2="600" y2="207" stroke="#4c1d95" stroke-width="3" marker-end="url(#purple-arrow)"/>
  <line x1="600" y1="275" x2="600" y2="297" stroke="#4c1d95" stroke-width="3" marker-end="url(#purple-arrow)"/>
  <line x1="600" y1="365" x2="600" y2="387" stroke="#4c1d95" stroke-width="3" marker-end="url(#purple-arrow)"/>
  <line x1="600" y1="455" x2="600" y2="507" stroke="#4c1d95" stroke-width="3" marker-end="url(#purple-arrow)"/>


  <!-- ==================== STAGE 1 (COMPLETED) ==================== -->
  <g filter="url(#card-shadow)">
    <rect x="300" y="25" width="600" height="70" rx="35" fill="#f3e8ff" stroke="#6d28d9" stroke-width="2.5" />
    <!-- Icon Circle -->
    <circle cx="348" cy="60" r="24" fill="#ffffff" stroke="#7c3aed" stroke-width="2"/>
    <!-- </> Code Icon -->
    <path d="M 342 53 L 334 60 L 342 67 M 354 53 L 362 60 L 354 67 M 350 50 L 346 70" fill="none" stroke="#4c1d95" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
    <!-- Text -->
    <text x="390" y="52" fill="#1e1b4b" font-size="18.5" font-weight="700">Data Ingestion &amp; Preprocessing</text>
    <text x="390" y="73" fill="#6b21a8" font-size="13" font-weight="500">Knee radiograph standardization, CLAHE enhancement, patient-disjoint splits</text>
  </g>


  <!-- ==================== STAGE 2 (COMPLETED) ==================== -->
  <g filter="url(#card-shadow)">
    <rect x="300" y="115" width="600" height="70" rx="35" fill="#f3e8ff" stroke="#6d28d9" stroke-width="2.5" />
    <!-- Icon Circle -->
    <circle cx="348" cy="150" r="24" fill="#ffffff" stroke="#7c3aed" stroke-width="2"/>
    <!-- Topological Graph / Complex Icon -->
    <circle cx="340" cy="144" r="3" fill="#4c1d95"/>
    <circle cx="356" cy="142" r="3" fill="#4c1d95"/>
    <circle cx="348" cy="158" r="3" fill="#4c1d95"/>
    <line x1="340" y1="144" x2="356" y2="142" stroke="#6d28d9" stroke-width="1.8"/>
    <line x1="356" y1="142" x2="348" y2="158" stroke="#6d28d9" stroke-width="1.8"/>
    <line x1="348" y1="158" x2="340" y2="144" stroke="#6d28d9" stroke-width="1.8"/>
    <!-- Text -->
    <text x="390" y="142" fill="#1e1b4b" font-size="18.5" font-weight="700">Topological Feature Extraction (TDA)</text>
    <text x="390" y="163" fill="#6b21a8" font-size="13" font-weight="500">Persistent Homology via GUDHI: H₀ components and H₁ bone architecture loops</text>
  </g>


  <!-- ==================== STAGE 3 (COMPLETED) ==================== -->
  <g filter="url(#card-shadow)">
    <rect x="300" y="205" width="600" height="70" rx="35" fill="#f3e8ff" stroke="#6d28d9" stroke-width="2.5" />
    <!-- Icon Circle -->
    <circle cx="348" cy="240" r="24" fill="#ffffff" stroke="#7c3aed" stroke-width="2"/>
    <!-- AI Brain / Chip Icon -->
    <rect x="339" y="231" width="18" height="18" rx="3" fill="none" stroke="#4c1d95" stroke-width="2"/>
    <path d="M 344 231 L 344 227 M 352 231 L 352 227 M 344 249 L 344 253 M 352 249 L 352 253 M 339 236 L 335 236 M 339 244 L 335 244 M 357 236 L 361 236 M 357 244 L 361 244" stroke="#4c1d95" stroke-width="1.8" stroke-linecap="round"/>
    <circle cx="348" cy="240" r="2.5" fill="#7c3aed"/>
    <!-- Text -->
    <text x="390" y="232" fill="#1e1b4b" font-size="18.5" font-weight="700">Deep Visual Backbone &amp; Multimodal Fusion</text>
    <text x="390" y="253" fill="#6b21a8" font-size="13" font-weight="500">ResNet-50 visual representation coupled with Late Probability Fusion</text>
  </g>


  <!-- ==================== STAGE 4 (COMPLETED) ==================== -->
  <g filter="url(#card-shadow)">
    <rect x="300" y="295" width="600" height="70" rx="35" fill="#f3e8ff" stroke="#6d28d9" stroke-width="2.5" />
    <!-- Icon Circle -->
    <circle cx="348" cy="330" r="24" fill="#ffffff" stroke="#7c3aed" stroke-width="2"/>
    <!-- Analytics / Verification Icon -->
    <rect x="339" y="322" width="18" height="16" rx="2" fill="none" stroke="#4c1d95" stroke-width="2"/>
    <path d="M 343 333 L 346 327 L 350 331 L 353 325" fill="none" stroke="#7c3aed" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <!-- Text -->
    <text x="390" y="322" fill="#1e1b4b" font-size="18.5" font-weight="700">Research Benchmarking &amp; Statistical Audit</text>
    <text x="390" y="343" fill="#6b21a8" font-size="13" font-weight="500">18-phase master pipeline, 5-fold cross-validation, McNemar test (p &lt; 10⁻²⁹)</text>
  </g>


  <!-- ==================== STAGE 5 (COMPLETED) ==================== -->
  <g filter="url(#card-shadow)">
    <rect x="300" y="385" width="600" height="70" rx="35" fill="#f3e8ff" stroke="#6d28d9" stroke-width="2.5" />
    <!-- Icon Circle -->
    <circle cx="348" cy="420" r="24" fill="#ffffff" stroke="#7c3aed" stroke-width="2"/>
    <!-- Monitor / Web App Icon -->
    <rect x="338" y="411" width="20" height="14" rx="2" fill="none" stroke="#4c1d95" stroke-width="2"/>
    <line x1="344" y1="429" x2="352" y2="429" stroke="#4c1d95" stroke-width="2" stroke-linecap="round"/>
    <line x1="348" y1="425" x2="348" y2="429" stroke="#4c1d95" stroke-width="2"/>
    <!-- Text -->
    <text x="390" y="412" fill="#1e1b4b" font-size="18.5" font-weight="700">Full-Stack Web Portal &amp; Inference API</text>
    <text x="390" y="433" fill="#6b21a8" font-size="13" font-weight="500">Interactive Streamlit diagnostic portal, Grad-CAM overlays &amp; FastAPI backend</text>
  </g>


  <!-- ==================== STAGE 6 (PENDING) ==================== -->
  <g filter="url(#card-shadow)">
    <rect x="300" y="515" width="600" height="70" rx="35" fill="#fff7ed" stroke="#f97316" stroke-width="2.5" />
    <!-- Icon Circle -->
    <circle cx="348" cy="550" r="24" fill="#ffffff" stroke="#ea580c" stroke-width="2"/>
    <!-- Rocket Icon -->
    <path d="M 353 540 C 353 540 357 544 355 550 L 344 561 C 342 559 341 556 341 554 L 346 547 C 348 543 353 540 353 540 Z" fill="#ea580c"/>
    <circle cx="350" cy="546" r="1.5" fill="#ffffff"/>
    <path d="M 342 558 L 337 561 L 340 556 Z" fill="#c2410c"/>
    <!-- Text -->
    <text x="390" y="542" fill="#1e1b4b" font-size="18.5" font-weight="700">Clinical PACS Integration &amp; Deployment</text>
    <text x="390" y="563" fill="#c2410c" font-size="13" font-weight="500">Hospital DICOM imaging pipeline, multi-center prospective validation &amp; edge deployment</text>
  </g>

</svg>
"""
    os.makedirs("results", exist_ok=True)
    svg_path = "results/implementation_status_diagram.svg"
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content.strip())
    print(f"SVG saved to {svg_path}")

if __name__ == "__main__":
    generate_svg()
