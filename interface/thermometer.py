import streamlit as st
import streamlit.components.v1 as components

def thermometer_slider(
    label: str,
    min_value: int = -5,
    max_value: int = 40,
    value: int = 4,
    step: int = 1,
    height: int = 360,
    key: str = "thermo_temp",
    color: str = "#E14F3D",
):
    """
    Vertical thermometer slider rendered in HTML.
    User drags mercury level inside thermometer to change value.
    Returns an int (or float if you change it) back to Streamlit.
    """
    # Streamlit component returns a value; we keep state in session_state
    if key not in st.session_state:
        st.session_state[key] = value

    # Use the last value as default (so it persists on reruns)
    value = st.session_state[key]

    if value < 4:
        temp_color = "#3498DB"
    elif value < 10:
        temp_color = "#27AE60"
    elif value < 25:
        temp_color = "#F39C12"
    else:
        temp_color = "#E14F3D"

    st.markdown(
        f"""
        <div style="text-align:center; font-size:20px; font-weight:700; color:{color}; margin-bottom:8px;">
            {value} °C
        </div>
        """,
        unsafe_allow_html=True
    )

    html = f"""
    <style>
      .thermo-wrap {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        display: flex;
        flex-direction: column;
        gap: 10px;
        user-select: none;
      }}
      .thermo-label {{
        font-size: 14px;
        font-weight: 600;
        color: #222;
      }}
      .thermo-row {{
        display:flex;
        align-items:center;
        gap: 16px;
      }}
      .thermo {{
        position: relative;
        width: 70px;
        height: 300px;
      }}
      .tube {{
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        width: 26px;
        height: 240px;
        top: 10px;
        border-radius: 18px;
        border: 3px solid #444;
        background: linear-gradient(180deg, #f7f7f7 0%, #ededed 100%);
        overflow: hidden;
      }}
      .bulb {{
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        bottom: 0px;
        width: 54px;
        height: 54px;
        border-radius: 50%;
        border: 3px solid #444;
        background: linear-gradient(180deg, #f7f7f7 0%, #eaeaea 100%);
        overflow: hidden;
      }}
      .mercury {{
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 0%;
        background: {color};
      }}
      .mercury-gloss {{
        position: absolute;
        left: 18%;
        width: 18%;
        top: 4%;
        bottom: 4%;
        border-radius: 99px;
        background: rgba(255,255,255,0.35);
      }}
      .bulb .mercury {{
        border-radius: 50%;
      }}
      .value-box {{
        min-width: 90px;
        font-size: 20px;
        font-weight: 800;
        color: {temp_color};
      }}
      .hint {{
        font-size: 12px;
        color: #666;
        margin-top: -6px;
      }}

      /* Invisible input overlay to capture drag inside thermometer */
      .overlay {{
        position:absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 10px;
        width: 26px;
        height: 240px;
        border-radius: 18px;
        cursor: ns-resize;
        background: rgba(0,0,0,0);
      }}

      /* Tick marks (optional) */
      .ticks {{
        position:absolute;
        left: calc(50% + 18px);
        top: 10px;
        height: 240px;
        width: 16px;
        display:flex;
        flex-direction: column;
        justify-content: space-between;
      }}
      .tick {{
        height: 2px;
        background: #555;
        border-radius: 2px;
        width: 10px;
        opacity: 0.7;
      }}
    </style>

    <div class="thermo-wrap">
      <div class="thermo-label">{label}</div>

      <div class="thermo-row">
        <div class="thermo" id="thermo">
          <div class="tube">
            <div class="mercury" id="tubeMercury"></div>
            <div class="mercury-gloss"></div>
          </div>

          <div class="bulb">
            <div class="mercury" id="bulbMercury"></div>
            <div class="mercury-gloss"></div>
          </div>

          <div class="overlay" id="dragArea" aria-label="drag temperature"></div>

          <div class="ticks">
            <div class="tick"></div>
            <div class="tick"></div>
            <div class="tick"></div>
            <div class="tick"></div>
            <div class="tick"></div>
            <div class="tick"></div>
          </div>
        </div>

        <div>
          <div class="value-box"><span id="valText">{value}</span> °C</div>
          <div class="hint">Drag inside the thermometer</div>
        </div>
      </div>
    </div>

    <script>
      const minV = {min_value};
      const maxV = {max_value};
      const step = {step};
      let value = {value};

      const tubeMercury = document.getElementById("tubeMercury");
      const bulbMercury = document.getElementById("bulbMercury");
      const valText = document.getElementById("valText");
      const dragArea = document.getElementById("dragArea");

      function clamp(x, a, b) {{
        return Math.max(a, Math.min(b, x));
      }}

      function snapToStep(v) {{
        const snapped = Math.round((v - minV) / step) * step + minV;
        return clamp(snapped, minV, maxV);
      }}

      function percentFromValue(v) {{
        return ((v - minV) / (maxV - minV)) * 100;
      }}

      function setUI(v) {{
        const pct = percentFromValue(v);
        tubeMercury.style.height = pct + "%";
        bulbMercury.style.height = "100%";
        valText.textContent = v;
      }}

      function emitValue(v) {{
        // Streamlit component protocol
        const out = {{ value: v }};
        window.parent.postMessage({{
          isStreamlitMessage: true,
          type: "streamlit:setComponentValue",
          value: out
        }}, "*");
      }}

      function updateFromClientY(clientY) {{
        const rect = dragArea.getBoundingClientRect();
        const y = clamp(clientY, rect.top, rect.bottom);
        const rel = (rect.bottom - y) / rect.height; // 0 bottom -> 1 top
        let v = minV + rel * (maxV - minV);
        v = snapToStep(v);
        value = v;
        setUI(value);
        emitValue(value);
      }}

      let dragging = false;

      dragArea.addEventListener("mousedown", (e) => {{
        dragging = true;
        updateFromClientY(e.clientY);
      }});

      window.addEventListener("mousemove", (e) => {{
        if (!dragging) return;
        updateFromClientY(e.clientY);
      }});

      window.addEventListener("mouseup", () => {{
        dragging = false;
      }});

      // Touch support
      dragArea.addEventListener("touchstart", (e) => {{
        dragging = true;
        updateFromClientY(e.touches[0].clientY);
      }}, {{passive:true}});

      window.addEventListener("touchmove", (e) => {{
        if (!dragging) return;
        updateFromClientY(e.touches[0].clientY);
      }}, {{passive:true}});

      window.addEventListener("touchend", () => {{
        dragging = false;
      }});

      // init UI
      setUI(value);
      emitValue(value);
    </script>
    """

    # Return value from component
    result = components.html(html, height=height)

    # result comes as dict {"value": x} or None
    if isinstance(result, dict) and "value" in result:
        st.session_state[key] = int(result["value"])

    return st.session_state[key]
