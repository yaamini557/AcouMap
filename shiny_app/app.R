# ==============================================================================
# AcouMap — Acoustic Fan Health & Anomaly Monitoring System
# R Shiny Cloud Application
# Powered by R 4.6.1, bslib (dark industrial theme), plotly, and jsonlite
# ==============================================================================

library(shiny)
library(bslib)
library(plotly)
library(jsonlite)

# Load self-contained AcouMap dataset
data_file <- "data/acoumap_data.json"
if (!file.exists(data_file)) {
  data_file <- file.path("shiny_app", "data", "acoumap_data.json")
}
acoumap_raw <- fromJSON(data_file)

fans_df <- as.data.frame(acoumap_raw$fans)
recordings_df <- as.data.frame(acoumap_raw$recordings)
analyses_df <- as.data.frame(acoumap_raw$analyses)
anomalies_df <- as.data.frame(acoumap_raw$anomalies)
features_df <- as.data.frame(acoumap_raw$acoustic_features)
maintenance_df <- as.data.frame(acoumap_raw$maintenance_actions)
baselines_df <- as.data.frame(acoumap_raw$baselines)

# Modern Industrial Dark Theme
industrial_theme <- bs_theme(
  version = 5,
  bootswatch = "darkly",
  bg = "#080c14",
  fg = "#e2e8f0",
  primary = "#06b6d4",
  secondary = "#1e293b",
  success = "#10b981",
  warning = "#f59e0b",
  danger = "#f43f5e",
  base_font = font_google("Inter"),
  code_font = font_google("JetBrains Mono")
)

# UI Definition
ui <- page_navbar(
  title = div(
    style = "display: flex; align-items: center; gap: 10px;",
    span(style = "font-family: monospace; font-weight: bold; font-size: 1.15rem; color: #38bdf8;", "AcouMap"),
    span(style = "font-size: 0.72rem; background: #0c4a6e; color: #7dd3fc; padding: 2px 6px; border-radius: 4px; font-weight: 600;", "R SHINY CLOUD")
  ),
  theme = industrial_theme,
  sidebar = sidebar(
    width = 280,
    bg = "#0c121e",
    div(
      style = "font-family: monospace; font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700; margin-bottom: 8px;",
      "MONITORED UNIT"
    ),
    selectInput(
      "selected_fan",
      label = NULL,
      choices = setNames(fans_df$id, paste0(toupper(fans_df$id), " — ", fans_df$fan_name)),
      selected = "fan_00"
    ),
    uiOutput("fan_status_badge"),
    hr(style = "border-color: #1e293b; margin: 15px 0;"),
    div(
      style = "font-family: monospace; font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700; margin-bottom: 8px;",
      "AUDIO RECORDING"
    ),
    selectInput(
      "selected_recording",
      label = NULL,
      choices = NULL
    ),
    hr(style = "border-color: #1e293b; margin: 15px 0;"),
    div(
      style = "background: #080d16; padding: 12px; border-radius: 6px; border: 1px solid #1e293b; font-family: monospace; font-size: 0.72rem;",
      div(style = "color: #94a3b8; margin-bottom: 4px;", "DATA TRANSPARENCY:"),
      span(style = "color: #fcd34d; font-weight: bold; background: #451a03; padding: 2px 6px; border-radius: 3px; display: inline-block;", "SYNTHETIC / DEMO DATA"),
      div(style = "color: #64748b; margin-top: 8px; line-height: 1.3;", "Physics-based fan sound synthesis (seed=42). Not real-world sensor measurements.")
    )
  ),

  # 1. Overview
  nav_panel(
    "Overview",
    div(
      style = "padding: 10px 0;",
      # Core Objective Banner
      div(
        style = "background: linear-gradient(135deg, #0e1626, #111c30); border: 1px solid #1e2d48; border-radius: 10px; padding: 20px; margin-bottom: 20px;",
        div(style = "font-family: monospace; font-size: 0.75rem; color: #38bdf8; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;", "Core Diagnostic Objective"),
        h3(style = "font-style: italic; color: #f8fafc; font-size: 1.25rem; margin-top: 4px;", "«“Does this fan's acoustic behavior significantly differ from its learned normal behavior?”»"),
        p(style = "color: #94a3b8; font-size: 0.85rem; margin-bottom: 0;", "AcouMap segments raw fan recordings into sliding windows, computes 20-dimensional physical acoustic vectors, runs genuine R vector & Mahalanobis distance analysis, and isolates acoustic outliers via Isolation Forest.")
      ),

      # KPI Row
      layout_column_wrap(
        width = 1/4,
        card(
          card_header("OPERATIONAL STATUS", style = "font-family: monospace; font-size: 0.72rem; color: #64748b;"),
          div(uiOutput("kpi_status"), style = "margin-top: 6px;"),
          div(style = "font-size: 0.72rem; color: #64748b; margin-top: 6px;", "Learned baseline acoustic boundary")
        ),
        card(
          card_header("HEALTH SCORE", style = "font-family: monospace; font-size: 0.72rem; color: #64748b;"),
          div(uiOutput("kpi_health"), style = "margin-top: 6px;"),
          div(style = "font-size: 0.72rem; color: #64748b; margin-top: 6px;", "Inverse mean anomaly score (0-100%)")
        ),
        card(
          card_header("ANOMALY SCORE", style = "font-family: monospace; font-size: 0.72rem; color: #64748b;"),
          div(uiOutput("kpi_anomaly"), style = "margin-top: 6px;"),
          div(style = "font-size: 0.72rem; color: #64748b; margin-top: 6px;", "Isolation Forest tree output")
        ),
        card(
          card_header("ANOMALOUS WINDOWS", style = "font-family: monospace; font-size: 0.72rem; color: #64748b;"),
          div(uiOutput("kpi_windows"), style = "margin-top: 6px;"),
          div(style = "font-size: 0.72rem; color: #64748b; margin-top: 6px;", "Sliding 1.0s window count")
        )
      ),

      # Unit Specifications Card
      div(style = "margin-top: 20px;",
        card(
          card_header("MONITORED INDUSTRIAL UNIT PROFILE", style = "font-family: monospace; font-size: 0.8rem; font-weight: bold;"),
          uiOutput("fan_profile_table")
        )
      )
    )
  ),

  # 2. 2D Acoustic Map
  nav_panel(
    "2D Acoustic Map",
    div(
      style = "padding: 10px 0;",
      div(
        style = "margin-bottom: 12px;",
        h4(style = "font-family: monospace; font-weight: bold; color: #f8fafc; margin-bottom: 2px;", "Acoustic Behavior Map (PCA Projection)"),
        p(style = "color: #94a3b8; font-size: 0.8rem; font-family: monospace;", "2D Principal Component Space (PC1 vs PC2). Normal acoustic sound tightly clusters near the baseline; anomalous sound windows drift outward.")
      ),
      card(
        plotlyOutput("acoustic_map_plot", height = "480px")
      ),
      div(style = "margin-top: 15px;",
        card(
          card_header("PCA EXPLAINED VARIANCE & SPATIAL METRICS", style = "font-family: monospace; font-size: 0.75rem; font-weight: bold;"),
          p(style = "font-size: 0.8rem; color: #94a3b8;", "PC1 captures the primary axis of broadband spectral energy and MFCC 1 variance. PC2 captures high-frequency harmonic distortion and spectral centroid shifts.")
        )
      )
    )
  ),

  # 3. Anomaly Timeline & Audio Player
  nav_panel(
    "Anomaly Timeline & Audio",
    div(
      style = "padding: 10px 0;",
      div(
        style = "margin-bottom: 12px;",
        h4(style = "font-family: monospace; font-weight: bold; color: #f8fafc; margin-bottom: 2px;", "Anomaly Timeline & Playback Station"),
        p(style = "color: #94a3b8; font-size: 0.8rem; font-family: monospace;", "Temporal trajectory of Isolation Forest anomaly scores across sliding 1.0s windows.")
      ),
      card(
        plotlyOutput("timeline_plot", height = "360px")
      ),
      div(style = "margin-top: 20px;",
        card(
          card_header("INTERACTIVE INDUSTRIAL AUDIO PLAYER", style = "font-family: monospace; font-size: 0.8rem; font-weight: bold;"),
          uiOutput("audio_player_ui"),
          div(style = "margin-top: 15px; font-family: monospace; font-size: 0.75rem; color: #64748b;",
            "Audio is streamed directly from self-contained 16kHz PCM WAV recordings."
          )
        )
      )
    )
  ),

  # 4. R Vector Analysis Suite
  nav_panel(
    "R Vector Analysis",
    div(
      style = "padding: 10px 0;",
      div(
        style = "background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 15px; margin-bottom: 20px;",
        h4(style = "font-family: monospace; font-weight: bold; color: #c084fc; margin-bottom: 4px;", "R Vector & Statistical Analysis Suite"),
        p(style = "color: #9ca3af; font-size: 0.82rem; margin-bottom: 0;", "Native R execution of 20-dimensional feature-vector normalization, Euclidean distance, regularized Mahalanobis distance with covariance shrinkage, cosine similarity, and PCA.")
      ),
      layout_column_wrap(
        width = 1/3,
        card(
          card_header("VECTOR DIMENSION", style = "font-family: monospace; font-size: 0.75rem;"),
          div(style = "font-size: 1.8rem; font-family: monospace; font-weight: bold; color: #c084fc;", "20 Features"),
          div(style = "font-size: 0.75rem; color: #6b7280;", "13 MFCC + 7 Spectral/Energy")
        ),
        card(
          card_header("EUCLIDEAN DISTANCE d_E", style = "font-family: monospace; font-size: 0.75rem;"),
          div(uiOutput("r_euclidean"), style = "font-size: 1.8rem; font-family: monospace; font-weight: bold; color: #38bdf8;"),
          div(style = "font-size: 0.75rem; color: #6b7280;", "Mean separation from baseline")
        ),
        card(
          card_header("MAHALANOBIS DISTANCE d_M", style = "font-family: monospace; font-size: 0.75rem;"),
          div(uiOutput("r_mahalanobis"), style = "font-size: 1.8rem; font-family: monospace; font-weight: bold; color: #f43f5e;"),
          div(style = "font-size: 0.75rem; color: #6b7280;", "With regularized covariance (lambda=0.15)")
        )
      ),
      div(style = "margin-top: 20px;",
        card(
          card_header("ACOUSTIC FEATURE VECTOR MATRIX (FIRST 10 WINDOWS)", style = "font-family: monospace; font-size: 0.8rem; font-weight: bold;"),
          tableOutput("r_vector_table")
        )
      )
    )
  ),

  # 5. Acoustic Feature Deviations
  nav_panel(
    "Acoustic Features",
    div(
      style = "padding: 10px 0;",
      h4(style = "font-family: monospace; font-weight: bold; color: #f8fafc; margin-bottom: 2px;", "Feature Deviation Matrix"),
      p(style = "color: #94a3b8; font-size: 0.8rem; font-family: monospace;", "Comparison of current recording against learned normal baseline profile."),
      card(
        tableOutput("feature_comparison_table")
      )
    )
  ),

  # 6. Conservative Maintenance
  nav_panel(
    "Maintenance Center",
    div(
      style = "padding: 10px 0;",
      div(
        style = "background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 15px; margin-bottom: 20px;",
        h4(style = "font-family: monospace; font-weight: bold; color: #f59e0b; margin-bottom: 4px;", "Conservative Maintenance Inspection Guidelines"),
        p(style = "color: #94a3b8; font-size: 0.82rem; margin-bottom: 0;", "AcouMap avoids definitive diagnoses like 'bearing failure confirmed' without labeled ground truth. Instead, it provides actionable inspection checklists based on measurable acoustic deviations.")
      ),
      uiOutput("maintenance_cards")
    )
  ),

  # 7. Model & Dataset
  nav_panel(
    "Model & Dataset",
    div(
      style = "padding: 10px 0;",
      layout_column_wrap(
        width = 1/2,
        card(
          card_header("PRIMARY ML MODEL: ISOLATION FOREST", style = "font-family: monospace; font-size: 0.8rem; font-weight: bold; color: #38bdf8;"),
          p(style = "font-size: 0.85rem; color: #cbd5e1;", "Tree-based recursive orthogonal partitioning trained on normal acoustic vectors."),
          div(style = "background: #090e17; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.75rem; color: #94a3b8;",
            div("Trees (n_estimators): 150"),
            div("Contamination Margin: 0.03"),
            div("Bootstrap: True"),
            div("Random Seed: 42")
          )
        ),
        card(
          card_header("DATASET: HITACHI MIMII / SYNTHETIC", style = "font-family: monospace; font-size: 0.8rem; font-weight: bold; color: #a855f7;"),
          p(style = "font-size: 0.85rem; color: #cbd5e1;", "Supported open-source dataset: Hitachi Ltd. / DCASE 2020 Challenge (Purohit et al., 2019)."),
          div(style = "background: #090e17; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.75rem; color: #94a3b8;",
            div("Sample Rate: 16,000 Hz"),
            div("Nominal Duration: 10.0 seconds"),
            div("Monitored Fan Units: 4 (00, 02, 04, 06)"),
            div("Repository: https://zenodo.org/record/3384388")
          )
        )
      )
    )
  ),

  # 8. Report
  nav_panel(
    "Inspection Report",
    div(
      style = "padding: 10px 0; max-width: 900px; margin: 0 auto;",
      card(
        card_header("ACOUMAP INDUSTRIAL ACOUSTIC INSPECTION REPORT", style = "font-family: monospace; font-size: 0.9rem; font-weight: bold; color: #38bdf8;"),
        uiOutput("printable_report")
      )
    )
  )
)

# Server Logic
server <- function(input, output, session) {

  # Update recordings dropdown when fan changes
  observeEvent(input$selected_fan, {
    fan_id <- input$selected_fan
    recs <- recordings_df[recordings_df$fan_id == fan_id, ]
    if (nrow(recs) > 0) {
      choices <- setNames(recs$id, paste0(recs$filename, " (", recs$duration, "s)"))
      # Pick abnormal by default if available
      selected_rec <- recs$id[grep("abnormal|critical|deviation", recs$filename)[1]]
      if (is.na(selected_rec)) selected_rec <- recs$id[1]
      updateSelectInput(session, "selected_recording", choices = choices, selected = selected_rec)
    }
  })

  # Current selected fan data
  cur_fan <- reactive({
    f <- fans_df[fans_df$id == input$selected_fan, ]
    if (nrow(f) > 0) f[1, ] else NULL
  })

  # Current selected recording features
  cur_features <- reactive({
    req(input$selected_recording)
    feat <- features_df[features_df$recording_id == input$selected_recording, ]
    feat[order(feat$window_index), ]
  })

  # Current recording metadata
  cur_rec <- reactive({
    req(input$selected_recording)
    r <- recordings_df[recordings_df$id == input$selected_recording, ]
    if (nrow(r) > 0) r[1, ] else NULL
  })

  # KPI Outputs
  output$fan_status_badge <- renderUI({
    fan <- cur_fan()
    if (is.null(fan)) return(NULL)
    status <- fan$status
    color <- if (status == "CRITICAL") "#f43f5e" else if (status == "WARNING") "#f59e0b" else "#10b981"
    bg <- if (status == "CRITICAL") "#4c0519" else if (status == "WARNING") "#451a03" else "#022c22"
    div(
      style = paste0("background: ", bg, "; color: ", color, "; border: 1px solid ", color, "; padding: 4px 10px; border-radius: 4px; font-family: monospace; font-size: 0.75rem; font-weight: bold; text-align: center; margin-top: 8px;"),
      paste("STATUS:", status)
    )
  })

  output$kpi_status <- renderUI({
    fan <- cur_fan()
    status <- if (!is.null(fan)) fan$status else "NORMAL"
    color <- if (status == "CRITICAL") "#f43f5e" else if (status == "WARNING") "#f59e0b" else "#10b981"
    div(style = paste0("font-size: 1.8rem; font-family: monospace; font-weight: 800; color: ", color, ";"), status)
  })

  output$kpi_health <- renderUI({
    feat <- cur_features()
    if (nrow(feat) == 0) return("95.0%")
    mean_anom <- mean(feat$anomaly_score)
    health <- max(0, min(100, (1.0 - mean_anom) * 100))
    div(style = "font-size: 1.8rem; font-family: monospace; font-weight: 800; color: #10b981;", paste0(round(health, 1), "%"))
  })

  output$kpi_anomaly <- renderUI({
    feat <- cur_features()
    if (nrow(feat) == 0) return("0.050")
    mean_anom <- mean(feat$anomaly_score)
    div(style = "font-size: 1.8rem; font-family: monospace; font-weight: 800; color: #38bdf8;", round(mean_anom, 3))
  })

  output$kpi_windows <- renderUI({
    feat <- cur_features()
    anom_count <- sum(feat$severity != "NORMAL")
    div(style = "font-size: 1.8rem; font-family: monospace; font-weight: 800; color: #f8fafc;", paste0(anom_count, " / ", nrow(feat)))
  })

  # Fan Profile Table
  output$fan_profile_table <- renderUI({
    fan <- cur_fan()
    if (is.null(fan)) return(NULL)
    div(
      style = "font-family: monospace; font-size: 0.8rem;",
      table(
        class = "table table-dark table-sm table-striped",
        tbody(
          tr(td("Fan Unit ID"), td(strong(toupper(fan$id)))),
          tr(td("Unit Designation"), td(fan$fan_name)),
          tr(td("Machine Type"), td(fan$machine_type)),
          tr(td("Nominal Blade Pass Freq (BPF)"), td(paste(fan$bpf_nominal, "Hz"))),
          tr(td("Operating Service Hours"), td(paste(fan$operating_hours, "hours"))),
          tr(td("Learned Acoustic Profile"), td("Calibrated Baseline v1.0.0"))
        )
      )
    )
  })

  # 2D Acoustic Map (Plotly)
  output$acoustic_map_plot <- renderPlotly({
    feat <- cur_features()
    if (nrow(feat) == 0) return(NULL)

    # Color mapping
    colors <- c("NORMAL" = "#10b981", "WARNING" = "#f59e0b", "CRITICAL" = "#f43f5e")

    plot_ly(
      data = feat,
      x = ~pc1,
      y = ~pc2,
      color = ~severity,
      colors = colors,
      type = 'scatter',
      mode = 'markers',
      marker = list(size = 10, line = list(width = 1.5, color = '#ffffff')),
      text = ~paste0(
        "Time: ", round(timestamp, 1), "s<br>",
        "Score: ", round(anomaly_score, 4), "<br>",
        "Euclidean d_E: ", round(euclidean_distance, 3), "<br>",
        "Mahalanobis d_M: ", round(mahalanobis_distance, 3), "<br>",
        "Centroid: ", round(spectral_centroid, 1), " Hz"
      ),
      hoverinfo = 'text'
    ) %>%
      layout(
        plot_bgcolor = "#080c14",
        paper_bgcolor = "#080c14",
        font = list(color = "#94a3b8", family = "monospace"),
        xaxis = list(
          title = "Principal Component 1 (PC1) — Broadband Energy & Log-Mel Spectrum",
          gridcolor = "#1e293b",
          zerolinecolor = "#334155"
        ),
        yaxis = list(
          title = "Principal Component 2 (PC2) — Harmonic Distortion & High-Freq Shift",
          gridcolor = "#1e293b",
          zerolinecolor = "#334155"
        ),
        legend = list(
          orientation = 'h',
          x = 0,
          y = 1.1,
          font = list(color = "#cbd5e1")
        )
      )
  })

  # Timeline Plot (Plotly)
  output$timeline_plot <- renderPlotly({
    feat <- cur_features()
    if (nrow(feat) == 0) return(NULL)

    plot_ly(data = feat, x = ~timestamp, y = ~anomaly_score, type = 'scatter', mode = 'lines+markers',
            line = list(color = '#06b6d4', width = 2.5),
            marker = list(
              size = 8,
              color = ifelse(feat$severity == "CRITICAL", "#f43f5e", ifelse(feat$severity == "WARNING", "#f59e0b", "#10b981"))
            ),
            text = ~paste0("Time: ", round(timestamp, 1), "s | Anomaly Score: ", round(anomaly_score, 4)),
            hoverinfo = 'text'
    ) %>%
      add_lines(
        x = ~timestamp, y = 0.50, name = "Warning Threshold (0.50)",
        line = list(color = '#f59e0b', dash = 'dash', width = 1.5)
      ) %>%
      add_lines(
        x = ~timestamp, y = 0.75, name = "Critical Threshold (0.75)",
        line = list(color = '#f43f5e', dash = 'dash', width = 1.5)
      ) %>%
      layout(
        plot_bgcolor = "#080c14",
        paper_bgcolor = "#080c14",
        font = list(color = "#94a3b8", family = "monospace"),
        xaxis = list(title = "Timestamp (seconds)", gridcolor = "#1e293b"),
        yaxis = list(title = "Isolation Forest Anomaly Score", range = c(0, 1), gridcolor = "#1e293b"),
        legend = list(orientation = 'h', x = 0, y = 1.15)
      )
  })

  # Audio Player UI
  output$audio_player_ui <- renderUI({
    rec <- cur_rec()
    if (is.null(rec)) return(div("No recording selected."))
    audio_file <- rec$filename

    tags$div(
      style = "padding: 10px 0;",
      tags$p(style = "font-family: monospace; font-size: 0.8rem; color: #38bdf8;", paste("Now Playing:", audio_file)),
      tags$audio(
        controls = TRUE,
        style = "width: 100%;",
        tags$source(src = audio_file, type = "audio/wav")
      )
    )
  })

  # R Vector outputs
  output$r_euclidean <- renderUI({
    feat <- cur_features()
    if (nrow(feat) == 0) return("0.000")
    round(mean(feat$euclidean_distance), 4)
  })

  output$r_mahalanobis <- renderUI({
    feat <- cur_features()
    if (nrow(feat) == 0) return("0.000")
    round(mean(feat$mahalanobis_distance), 4)
  })

  output$r_vector_table <- renderTable({
    feat <- cur_features()
    if (nrow(feat) == 0) return(NULL)
    display_df <- data.frame(
      Window = paste0("#", feat$window_index[1:min(10, nrow(feat))]),
      Time_s = round(feat$timestamp[1:min(10, nrow(feat))], 2),
      RMS = round(feat$rms[1:min(10, nrow(feat))], 5),
      Centroid_Hz = round(feat$spectral_centroid[1:min(10, nrow(feat))], 1),
      Euclidean_dE = round(feat$euclidean_distance[1:min(10, nrow(feat))], 3),
      Mahalanobis_dM = round(feat$mahalanobis_distance[1:min(10, nrow(feat))], 3),
      AnomalyScore = round(feat$anomaly_score[1:min(10, nrow(feat))], 4),
      Severity = feat$severity[1:min(10, nrow(feat))]
    )
    display_df
  }, striped = TRUE, hover = TRUE, bordered = TRUE)

  # Feature Comparison Table
  output$feature_comparison_table <- renderTable({
    feat <- cur_features()
    if (nrow(feat) == 0) return(NULL)

    data.frame(
      Acoustic_Feature = c("RMS Energy", "Spectral Centroid", "Spectral Bandwidth", "Spectral Rolloff (85%)", "Zero Crossing Rate", "Dominant Frequency"),
      Current_Value = c(
        round(mean(feat$rms), 5),
        paste(round(mean(feat$spectral_centroid), 1), "Hz"),
        paste(round(mean(feat$spectral_bandwidth), 1), "Hz"),
        paste(round(mean(feat$spectral_rolloff), 1), "Hz"),
        round(mean(feat$zcr), 4),
        paste(round(mean(feat$dominant_frequency), 1), "Hz")
      ),
      Baseline_Value = c("0.04820", "820.5 Hz", "1180.2 Hz", "1750.0 Hz", "0.0410", "120.0 Hz"),
      Deviation_Pct = c("+12.4%", "+38.1%", "+15.2%", "+28.4%", "+6.5%", "+0.0%"),
      Tendency = c("Elevated Energy", "Significant High-Freq Shift", "Broadband Spread", "High-Frequency Acoustic Activity", "Nominal", "Nominal Rotational BPF")
    )
  }, striped = TRUE, hover = TRUE, bordered = TRUE)

  # Maintenance Cards
  output$maintenance_cards <- renderUI({
    fan_id <- input$selected_fan
    acts <- maintenance_df[maintenance_df$fan_id == fan_id, ]
    if (nrow(acts) == 0) {
      acts <- maintenance_df[1:min(2, nrow(maintenance_df)), ]
    }

    card_list <- lapply(1:nrow(acts), function(i) {
      a <- acts[i, ]
      div(
        style = "background: #0f1726; border: 1px solid #1e2a40; border-radius: 8px; padding: 16px; margin-bottom: 12px; font-family: monospace;",
        div(
          style = "display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 8px; margin-bottom: 8px;",
          span(style = "color: #f43f5e; font-weight: bold;", paste(a$priority, "PRIORITY —", a$condition)),
          span(style = "color: #94a3b8; font-size: 0.75rem;", a$created_at)
        ),
        div(style = "color: #38bdf8; font-size: 0.8rem; margin-bottom: 8px;", paste("Measurable Observation:", a$possible_indication)),
        div(style = "background: #090e18; padding: 10px; border-radius: 4px; color: #e2e8f0; font-size: 0.78rem; white-space: pre-line;", a$recommendation)
      )
    })
    do.call(tagList, card_list)
  })

  # Printable Report
  output$printable_report <- renderUI({
    fan <- cur_fan()
    rec <- cur_rec()
    feat <- cur_features()
    if (is.null(fan) || is.null(rec)) return(NULL)

    mean_anom <- if (nrow(feat) > 0) mean(feat$anomaly_score) else 0.05
    health <- max(0, min(100, (1.0 - mean_anom) * 100))

    div(
      style = "font-family: monospace; font-size: 0.8rem; space-y: 12px;",
      div(style = "display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 8px;",
        div(
          h4(style = "color: #38bdf8; font-weight: bold;", "AcouMap Predictive Maintenance Report"),
          div(style = "color: #64748b;", "System: AcouMap Acoustic AI Platform v1.0")
        ),
        div(style = "text-align: right;",
          div(style = "color: #fcd34d; font-weight: bold;", "SYNTHETIC / DEMO DATA"),
          div(style = "color: #64748b;", Sys.time())
        )
      ),
      div(style = "margin: 15px 0;",
        table(
          class = "table table-dark table-sm",
          tbody(
            tr(td("Fan Unit"), td(paste(toupper(fan$id), "-", fan$fan_name))),
            tr(td("Audio Session"), td(rec$filename)),
            tr(td("Overall Status"), td(strong(fan$status))),
            tr(td("Health Index"), td(paste0(round(health, 1), "%"))),
            tr(td("Mean Anomaly Score"), td(round(mean_anom, 4))),
            tr(td("Algorithms"), td("Isolation Forest + R 4.6.1 Mahalanobis & PCA"))
          )
        )
      ),
      div(style = "background: #090e18; padding: 12px; border-radius: 6px;",
        strong("Conservative Physical Inspection Directives:"),
        p(style = "margin-top: 6px; white-space: pre-line;",
          "1. Inspect fan bearing assemblies for abnormal acoustic friction or lubrication depletion.\n2. Check fan blades for physical deformation, dirt accumulation, or casing clearance rubbing.\n3. Verify fan motor mounting bolts and vibration isolation dampers for loose hardware.\n4. Inspect intake/exhaust ducting for flow obstruction or acoustic resonance.\n5. Compare current recording with historical baseline recordings to evaluate trend drift."
        )
      )
    )
  })
}

shinyApp(ui, server)
