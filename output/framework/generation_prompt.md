Use case: infographic-diagram, precise scientific diagram edit.
Asset type: polished wide academic machine-learning architecture figure, English labels, about 3840 x 2048 pixels.
Image 1 is the EDIT TARGET and its macro layout is the primary invariant. Refine and scientifically correct the supplied MacDiff text/skeleton framework image using the local-code specification below. Keep the same landscape aspect ratio near 1.88:1 and same four regional positions: pale sage green text-generation region across top-left ~75% width; narrow pale gray skeleton-diffusion column at far right; small rose-pink text-diffusion panel center-right in middle row; wide pale ice-blue skeleton-encoder band along the whole bottom. Keep the visualization block at middle-left and the two noise-sampling blocks in the middle-left gap. Retain skeleton thumbnails, stacked feature-token glyphs, encoder trapezoids and left-to-right reading direction. Add only necessary small submodules. Do NOT radically rearrange this into a generic three-column pipeline.

Style:
A publication-quality, crisp flat vector-looking figure on white background. Restrained pastel sage, warm peach, ice blue, blush pink. Thin dark navy strokes with consistent arrowheads; tidy orthogonal routing, generous whitespace, alignment and visual hierarchy. Clean professional sans-serif body labels and italic serif mathematics. Headers visibly larger than captions. Avoid dense microscopic captions, overlapping labels, decorative gradients, heavy shadows, photorealism, watermark. Straight/orthogonal arrows must clearly terminate at the module they belong to. Use exact readable English below. Represent math with well-rendered subscripts/superscripts or equally readable plain text. The output should be one finished diagram, no explanatory surrounding page.

Scientific structure to draw:
1) TOP GREEN region title "Text Generation and Remap".
Left to right: skeleton front/side visualization of the SAME person -> "Prompt" -> "VLM" -> "Frozen CLIP Text Encoder" -> feature glyph labeled "Cached sentence features F_i" with caption "1 global + 6 body regions; 7 x 512; RMS normalized" -> trainable box "Residual Remap" with subtitle "MLP + skip + RMS" -> online feature glyph labeled "Online text R_phi(F_i)".
Six regions: head, torso, left arm, right arm, left leg, right leg. Put this list as a short one-line caption, not six unnecessary boxes.
Clearly mark VLM/CLIP and cache as offline/frozen. The remap is trainable. Global feature is one green cube and local features are six colored stacked cubes as in the reference.
Put a compact small box under the Remap within the lower edge of the green region: "Per-sample Target Bank B_i". Grey thin initialization arrow from F_i to bank, labeled "Initialize: B_i = F_i". A green dashed update arrow from Remap to bank, labeled "After successful optimizer step". Formula inside/beside bank:
"B_i <- 0.9 B_i + 0.1 R_phi(F_i)"
Small note "post-step weights; no gradient".
IMPORTANT: this is a recursive blend of OLD saved sample target and current online remap, NOT a blend of permanently fixed CLIP and current remap every forward. Do not reproduce the old alpha mixer in the reference. Bank output is labeled "stop-grad target". The bank read goes ONLY to text forward diffusion, not directly to the T->S conditioning input.
Online remap goes to T->S as conditioning with label "global + local text". Put small "L_uni^text" near online LOCAL content features only.

2) MIDDLE LEFT:
Keep yellow "Visualization" block and its arrow from the skeleton input to offline VLM-rendering thumbnails above, labeled "Front / side views (same person)".
Upper peach noise box: "Text Forward Diffusion", with compact formula "F_t = sqrt(a_t) B_i + sqrt(1-a_t) epsilon_t"; it receives stop-gradient B_i, then produces a stacked noisy-text feature glyph "Noisy text F_t". Its timestep label is "t_t". It goes right into the pink text-diffusion panel. Use legible simple typography if a_bar is hard; do not truncate equations.
Lower peach noise box: "Skeleton Forward Diffusion", formula "x_t = sqrt(a_t) x_0 + sqrt(1-a_t) epsilon_s". Clean normalized skeleton x_0 branches to this box. It produces skeleton thumbnail "Noisy skeleton x_t". Its timestep label is "t_s". This feeds BOTH independent native and T->S skeleton decoders in the right column, through a shared-input bus labeled "Same x_t, t_s, epsilon_s, mask".

3) MIDDLE ROSE PINK region:
Title "Skeleton-to-Text Diffusion".
Noisy text F_t -> "Text Embed + Structure" -> blue rounded box "S->T Diffusion Decoder" -> "Linear Head 512 -> 512" -> "L_S2T".
Decoder subtitle "5 blocks; hidden 512".
Skeleton condition arrow from bottom encoder GLOBAL pooled token + 75 visible tokens into this decoder: label "Skeleton memory: 76 x 256". The decoder has a little legible inside-caption "Skeleton cross-attention -> AdaLN -> Text self-attention -> MLP".
t_t goes into decoder; it is sampled independently of t_s. Output is epsilon_t prediction. Loss is "MSE over valid 7 text vectors", not text token CE.
Small note "No final LayerNorm" by output head. The saved target bank is detached BEFORE corruption. No arrow or gradient from S->T loss to remap/cache. Direct gradients to skeleton encoder through skeleton memory exist.
Allow small arrows within panel; main caption enough; do not build dozens of microblocks.

4) RIGHT PALE GRAY TALL column:
Title "Skeleton Diffusion"; subtitle "Independent decoders (no-share)".
CRITICAL: ADD the missing native branch by placing TWO separate compact blue decoder cards stacked vertically within this existing far-right column, rather than moving the column or expanding over the bottom band.
UPPER card "Native Skeleton Decoder", subtitle "3 blocks; hidden 256". Receives noisy x_t, t_s, and bottom skeleton-encoder conditioning via small box "Restore + global fill", caption "750 x 256". This condition fills masked positions with global pooled features and keeps local visible embeddings. A small note "condition dropout 0.1". Output "LN + Linear 256 -> 12" -> "L_native".
LOWER card "T->S Skeleton Decoder", subtitle "3 blocks; hidden 256". Receives the exact same noisy skeleton x_t and t_s as native, but online remapped GLOBAL/LOCAL TEXT conditioning from the top green feature glyph. Small note "local + person/region embeddings". Its block can be represented as "Text cross-attention -> AdaLN -> MSA -> MLP", then "LN + Linear 256 -> 12" -> "L_T2S".
These two decoders are independently parameterized; do not mark them shared. Native uses skeleton conditioning, T->S uses text conditioning. There must be no skeleton-encoder conditioning arrow to T->S.
Both skeleton losses are "Masked-patch noise MSE". There are three denoising loss labels total: L_native, L_T2S, L_S2T, not only the original two.
Maintain visual balance and adequate text size within column.

5) BOTTOM ICE-BLUE band:
Title/caption "Masked Skeleton Representation Learning".
Keep the sequence of frame thumbnails at left, then "Patch Embed", then "Random Masking", then visible patch token glyphs, then yellow encoder trapezoid "Skeleton Encoder", then a global green cube + visible local-feature stacked glyphs at right, as in reference.
Input label "Skeleton sequence x_0; person 0". Encoder input path includes a short annotation "encoder-only Gaussian noise (sigma = 0.005, raw coordinates)" before model normalization. Annotate "Crop / resize: 120 frames; A normalization". Clean input branches separately to skeleton forward diffusion; encoder-added Gaussian noise must not contaminate the clean diffusion target.
Patch Embed subtitle "4 frames x 1 joint -> 256". Mask subtitle "90% masked". Visible patch label "75 / 750 patches". Encoder subtitle "8 blocks; hidden 256". Output label "Global pooled + visible local features".
Small branch from visible local features to "L_uni^skel", replacing the reference's vague "L_SSL". It is a token uniformity loss, not contrastive learning.
Route output condition to both Native decoder (through Restore + global fill) and S->T decoder (pooled + visible tokens); not to T->S.
Include a small optional evaluation-only caption at the bottom edge: "Linear probe: frozen skeleton encoder -> BN + Linear(6400, 60); no text at evaluation". This should be a small caption, not a new large panel and not a training loss.

6) LEGEND and objective, unobtrusively placed in bottom margin:
Use dark solid arrows for forward inputs, blue solid arrows for conditioning, green dashed arrows for target-bank update. Use a clearly labeled stop-gradient symbol on bank read. To avoid misleading reverse arrows, do NOT use red gradient arrows crossing unrelated features. If gradient routes are drawn, red dashed arrows must only indicate L_native/L_S2T -> encoder and L_T2S/L_uni^text -> remap; no L_T2S -> encoder and no L_S2T -> remap. A compact note is preferable: "Encoder trained by native + S->T + skeleton uniformity".
Objective EXACT:
"L = L_native + L_T2S + 0.1 L_S2T + 0.02 L_uni^skel + 0.02 L_uni^text"
No contrastive term, parameter EMA or fake supervision.

Preserve the reference's overall region placements and diagram identity aggressively, while replacing ambiguous/dense text and wrong arrows. Correct code dataflow is more important than copying inaccurate wiring. Every module name should be readable. Prefer a few restrained small callouts to excessive tiny formulas. The edit must clearly show the online remap and saved bank as different objects, and the native decoder must remain visible.

--- Revision: wiring correction ---
Edit this generated scientific architecture figure, keeping ALL main panel positions, colors, font style, thumbnails, dimensions, module placement and the attractive publication layout. This is a WIRING CORRECTION only, not a redesign. Preserve the four regional layout and current headers, formula, decoder depths and sizes.

Delete misleading extra cross-region lines and re-route inter-region arrows cleanly using the following EXACT directed graph. All inter-region arrows not in this list must be absent. Use distinct orthogonal routes that do not join merely because they cross; all arrowheads terminate on their intended receiving box.

A. Input skeleton -> Visualization. Visualization -> front/side thumbnails -> VLM -> frozen CLIP -> Cached sentence features F_i -> Residual Remap -> Online text R_phi(F_i). Keep this offline/top row sequence.
B. Cached F_i -> Per-sample Target Bank, grey initialization arrow. Residual Remap -> bank, green dashed POST-STEP update arrow. Keep recursive equation B_i <- 0.9 B_i + 0.1 R_phi(F_i).
C. The ONLY read-output of Target Bank B_i MUST go through a clearly labeled stop-gradient symbol INTO THE LEFT EDGE OF "Text Forward Diffusion" in the middle-left. Route this read along the lower edge of green panel, LEFTWARD then downward into the text noise box. It must NEVER go directly into Text Embed or S->T decoder. Remove current erroneous bank-to-S->T upper line. Also remove any Visualization-to-bank or Visualization-to-text-noise arrows.
D. Text Forward Diffusion -> Noisy text F_t glyph -> Text Embed + Structure -> S->T Decoder -> Linear512->512 -> L_S2T. t_t enters the S->T decoder as a separately labeled timestep arrow; do not connect that timestep arrow to bank.
E. Clean normalized skeleton x_0 -> Skeleton Forward Diffusion -> Noisy skeleton x_t glyph. From noisy skeleton x_t, draw one DARK input bus splitting into Native Skeleton Decoder and T->S Skeleton Decoder. Both take x_t and t_s. Epsilon_s and mask are shared LOSS TARGETS, not network conditioning inputs; remove epsilon_s from the list of decoder input labels. Add small bus note "Same x_t and t_s; shared noise target / mask".
F. Online text R_phi(F_i) -> T->S Decoder condition input ONLY, with BLUE arrow and inline caption "global + local text; local + person/region embeddings". Route this from top-right green glyph down in a narrow channel along the left edge of the right gray panel and into the LOWER decoder card. It must NOT touch or join the Native Decoder condition input and must NOT join the skeleton-memory bus.
G. Bottom skeleton encoder -> Global pooled + visible local features glyph -> Skeleton memory76x256 box below pink -> S->T Decoder (BLUE conditioning arrow). This skeleton memory ALSO may branch through Restore + global fill750x256 to Native Decoder, using a distinct blue route. It must NEVER connect to T->S. There is currently an erroneous blue line from skeleton memory to the bottom edge of T->S, labeled local+person/region; remove that entire connection. That caption belongs to incoming ONLINE TEXT, not skeleton memory.
H. Native decoder head -> L_native. T->S decoder head -> L_T2S. Keep two separate decoders.
I. Online local text features -> L_uni^text, ordinary DARK forward loss arrow directed FROM FEATURES TO LOSS. Bottom visible skeleton features -> L_uni^skel, ordinary DARK forward loss arrow directed FROM FEATURES TO LOSS. Remove ALL red dashed gradient arrows across the figure, including the incorrect loop inside pink. Keep ONLY the small footer text "Encoder trained by native + S->T + skeleton uniformity". The diagram communicates gradient facts by this accurate caption; no confusing reverse arrows needed. Remove red-gradient item from legend. Legend should show forward input, conditioning, target bank update(no gradient), stop-gradient only.

Fix two local captions:
- Native decoder internal white box should read "AdaLN -> MSA -> AdaLN -> MLP". Remove the spurious "Skeleton self-attention" extra stage.
- Encoder added Gaussian noise is in RAW coordinates BEFORE A normalization. In bottom band, keep small preprocessing labels but show unambiguous inline ordering "Crop120 -> Gaussian noise -> A normalization -> Patch Embed" for encoder input; CLEAN diffusion branch is "Crop120 -> A normalization" without small Gaussian. Use small tidy boxes or a two-line caption with a correctly ordered arrow; preserve bottom Embed-Masking-Encoder macro positions.

Small thumbnail accuracy: show one retained person in bottom sequence and noisy-skeleton glyph since this implementation uses person0. The top front/side thumbnails may show two colors but label them as two views of the same person, not two people.

Do not change objective weights:
L = L_native + L_T2S + 0.1 L_S2T + 0.02 L_uni^skel + 0.02 L_uni^text.
Make all equations legible, no new arrows/losses/components outside the listed graph. Maximize readability, retain the existing macro layout, and give this diagram at large high resolution.