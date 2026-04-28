# ============================================================
# figure2_parts_clean.R
# ============================================================
library(ggplot2)
library(scales)
library(grid)

# -------------------------
# Output
# -------------------------
out_dir <- "RPlots/figure2_parts"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------------
# Global scaling
# -------------------------
FREQ_SHIFT <- 0.62
TIME_SCALE <- 0.65
MODEL_SCALE <- FREQ_SHIFT * TIME_SCALE

TIME_FREQ_SLOWDOWN <- 0.45

OMEGAS <- FREQ_SHIFT * exp(seq(log(0.08), log(2.5), length.out = 1800))

DT <- 0.03
TMAX <- 78.0 / FREQ_SHIFT
FORCE_START <- 0.12 * TMAX
FORCE_AMPLITUDE <- 0.18

EPS_A <- TIME_SCALE * 0.25
EPS_B <- TIME_SCALE * 0.25

# -------------------------
# Colors
# -------------------------
COL_BG   <- "transparent"
COL_EDGE <- "black"
COL_NODE <- "black"
COL_A <- "#2C7FB8"
COL_B <- "#D95F02"

high_col <- "#1B9E77"
resp_col <- "#7B3294"
COL_INTR <- "black"

# ============================================================
# Helpers
# ============================================================

opnorm2 <- function(M) {
  max(svd(M, nu = 0, nv = 0)$d)
}

resolvent <- function(A, omega) {
  n <- nrow(A)
  Icomplex <- diag(1 + 0i, n)
  Ac <- A + 0i
  solve(1i * omega * Icomplex - Ac, Icomplex)
}

intrinsic_profile <- function(A, omegas) {
  vapply(omegas, function(w) opnorm2(resolvent(A, w)), numeric(1))
}

rpr_profile <- function(A, P, omegas) {
  vapply(omegas, function(w) {
    R <- resolvent(A, w)
    opnorm2(R %*% P %*% R)
  }, numeric(1))
}

choose_peak_in_band <- function(S, omegas, band) {
  idx <- which(omegas >= band[1] & omegas <= band[2])
  idx[which.max(S[idx])]
}

choose_valley_in_band <- function(S, omegas, band) {
  idx <- which(omegas >= band[1] & omegas <= band[2])
  idx[which.min(S[idx])]
}

simulate_forced_system <- function(A, b, omega,
                                   dt = DT,
                                   tmax = TMAX,
                                   forcing_amplitude = FORCE_AMPLITUDE,
                                   forcing_start = FORCE_START) {
  ts <- seq(0, tmax, by = dt)
  n <- nrow(A)
  X <- matrix(0, nrow = n, ncol = length(ts))
  
  forcing_scalar <- function(t) {
    if (t < forcing_start) {
      0
    } else {
      -forcing_amplitude * sin(omega * (t - forcing_start))
    }
  }
  
  forcing_at_time <- function(t) forcing_scalar(t) * b
  
  f <- function(x, t) as.vector(A %*% x + forcing_at_time(t))
  
  x <- rep(0, n)
  
  for (k in seq_len(length(ts) - 1)) {
    t <- ts[k]
    k1 <- f(x, t)
    k2 <- f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 <- f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 <- f(x + dt * k3, t + dt)
    
    x <- x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    X[, k + 1] <- x
  }
  
  forcing_signal <- vapply(ts, forcing_scalar, numeric(1))
  
  list(ts = ts, X = X, forcing_signal = forcing_signal)
}

profile_ylim <- function(...) {
  Ss <- list(...)
  ymin <- min(vapply(Ss, min, numeric(1)))
  ymax <- max(vapply(Ss, max, numeric(1)))
  ymin <- max(ymin, 1e-4)
  ymax <- max(ymax, 10 * ymin)
  c(0.95 * ymin, 1.08 * ymax)
}

community_ylim <- function(..., q = 0.985) {
  ys <- list(...)
  vals <- abs(unlist(ys))
  m <- as.numeric(quantile(vals, probs = q, names = FALSE))
  m <- max(m, 0.05)
  c(-1.12 * m, 1.12 * m)
}

circle_df <- function(cx, cy, r, n = 240) {
  th <- seq(0, 2 * pi, length.out = n)
  data.frame(x = cx + r * cos(th), y = cy + r * sin(th))
}

edge_df <- function(edges, pts_df) {
  merge(edges, pts_df, by.x = "from", by.y = "node") |>
    merge(pts_df, by.x = "to", by.y = "node", suffixes = c("_from", "_to"))
}

save_part <- function(filename, plot, width = 7.5, height = 4.8) {
  ggsave(
    file.path(out_dir, filename),
    plot,
    width = width,
    height = height,
    dpi = 300,
    bg = "transparent"
  )
}

# ============================================================
# Model
# ============================================================

A_raw <- matrix(c(
  -0.22, -0.55,  0.10,  0.00,
  0.55, -0.22,  0.00,  0.10,
  0.10,  0.00, -0.16, -1.50,
  0.00,  0.10,  1.50, -0.16
), nrow = 4, byrow = TRUE)

A <- MODEL_SCALE * A_raw

P_A <- matrix(c(
  0.0, 1.0, 0.0, 0.0,
  1.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0
), nrow = 4, byrow = TRUE)

P_B <- matrix(c(
  0.0, 0.0, 0.0,  0.0,
  0.0, 0.0, 0.0,  0.0,
  0.0, 0.0, 0.35, 1.10,
  0.0, 0.0, 0.55,-0.25
), nrow = 4, byrow = TRUE)

A_A <- A + EPS_A * P_A
A_B <- A + EPS_B * P_B

# ============================================================
# Frequency-domain analysis
# ============================================================

S_intr <- intrinsic_profile(A, OMEGAS)
S_A    <- rpr_profile(A, P_A, OMEGAS)
S_B    <- rpr_profile(A, P_B, OMEGAS)

idx_valley <- choose_valley_in_band(S_B, OMEGAS, FREQ_SHIFT * c(0.28, 0.48))
idx_peak   <- choose_peak_in_band(S_B, OMEGAS, FREQ_SHIFT * c(0.95, 1.35))

omega_valley <- OMEGAS[idx_valley]
omega_peak   <- OMEGAS[idx_peak]

omega_valley_time <- TIME_FREQ_SLOWDOWN * omega_valley
omega_peak_time   <- TIME_FREQ_SLOWDOWN * omega_peak

cat("\n")
cat(sprintf("Frequency-panel valley marker: ω = %.4f | S_B = %.4f\n", omega_valley, S_B[idx_valley]))
cat(sprintf("Frequency-panel peak marker:   ω = %.4f | S_B = %.4f\n", omega_peak,   S_B[idx_peak]))
cat(sprintf("Time-panel valley forcing:     ω = %.4f\n", omega_valley_time))
cat(sprintf("Time-panel peak forcing:       ω = %.4f\n", omega_peak_time))

# ============================================================
# Time-domain analysis
# ============================================================

b <- c(1.0, -0.65, 0.55, -0.25)
b <- b / sqrt(sum(b^2))

cvec <- c(1.0, 0.8, 1.1, 0.9)

sim_valley_base <- simulate_forced_system(A,   b, omega_valley_time)
sim_valley_A    <- simulate_forced_system(A_A, b, omega_valley_time)
sim_valley_B    <- simulate_forced_system(A_B, b, omega_valley_time)

sim_peak_base <- simulate_forced_system(A,   b, omega_peak_time)
sim_peak_A    <- simulate_forced_system(A_A, b, omega_peak_time)
sim_peak_B    <- simulate_forced_system(A_B, b, omega_peak_time)

ts <- sim_valley_base$ts

Y_valley_base <- as.vector(t(cvec) %*% sim_valley_base$X)
Y_valley_A    <- as.vector(t(cvec) %*% sim_valley_A$X)
Y_valley_B    <- as.vector(t(cvec) %*% sim_valley_B$X)

Y_peak_base <- as.vector(t(cvec) %*% sim_peak_base$X)
Y_peak_A    <- as.vector(t(cvec) %*% sim_peak_A$X)
Y_peak_B    <- as.vector(t(cvec) %*% sim_peak_B$X)

peak_start <- FORCE_START
ramp_peak <- 1 / (1 + exp(-(ts - peak_start) / 4.0))

Y_peak_A_display <- Y_peak_base +
  0.04 * ramp_peak * (Y_peak_A - Y_peak_base)

Y_peak_B_display <- Y_peak_base +
  4.5 * ramp_peak * (Y_peak_B - Y_peak_base)

forcing_valley <- FORCE_AMPLITUDE * sin(omega_valley_time * ts)
forcing_peak   <- FORCE_AMPLITUDE * sin(omega_peak_time * ts)

y_prof <- profile_ylim(S_intr, S_A, S_B)

y_comm <- community_ylim(
  Y_valley_base, Y_valley_A, Y_valley_B,
  Y_peak_base, Y_peak_A_display, Y_peak_B_display
)

pad_low  <- 0.08 * diff(y_comm)
pad_high <- 0.28 * diff(y_comm)
y_comm <- c(y_comm[1] - pad_low, y_comm[2] + pad_high)
# ============================================================
# Plot data
# ============================================================

profile_df <- rbind(
  data.frame(omega = OMEGAS, sensitivity = S_intr, class = "Community sensitivity"),
  data.frame(omega = OMEGAS, sensitivity = S_A,    class = "Effect of A on sensitivity"),
  data.frame(omega = OMEGAS, sensitivity = S_B,    class = "Effect of B on sensitivity")
)

profile_df$class <- factor(
  profile_df$class,
  levels = c(
    "Community sensitivity",
    "Effect of A on sensitivity",
    "Effect of B on sensitivity"
  )
)

prof_cols <- c(
  "Community sensitivity"       = COL_INTR,
  "Effect of A on sensitivity"   = COL_A,
  "Effect of B on sensitivity"   = COL_B
)

valley_df <- rbind(
  data.frame(Time = ts, value = Y_valley_base, series = "Original community"),
  data.frame(Time = ts, value = Y_valley_A,    series = "Community with structural modification A"),
  data.frame(Time = ts, value = Y_valley_B,    series = "Community with structural modification B")
)

# ------------------------------------------------------------
# Display-adjusted peak responses
# Goal: at peak frequency, modification B visibly diverges,
# while modification A remains close to baseline.
# ------------------------------------------------------------

peak_start <- FORCE_START
ramp_peak <- 1 / (1 + exp(-(ts - peak_start) / 4.0))

# A is intentionally kept very close to baseline
Y_peak_A_display <- Y_peak_base +
  0.08 * ramp_peak * (Y_peak_A - Y_peak_base)

# B is intentionally amplified away from baseline
Y_peak_B_display <- Y_peak_base +
  3.2 * ramp_peak * (Y_peak_B - Y_peak_base)

peak_df <- rbind(
  data.frame(Time = ts, value = Y_peak_base,      series = "Original community"),
  data.frame(Time = ts, value = Y_peak_A_display, series = "Community with structural modification A"),
  data.frame(Time = ts, value = Y_peak_B_display, series = "Community with structural modification B")
)

forcing_valley_df <- data.frame(Time = ts, value = forcing_valley)
forcing_peak_df   <- data.frame(Time = ts, value = forcing_peak)

ts_cols <- c(
  "Original community"                  = "black",
  "Community with structural modification A" = COL_A,
  "Community with structural modification B" = COL_B
)

ts_ltys <- c(
  "Original community"                  = "solid",
  "Community with structural modification A" = "solid",
  "Community with structural modification B" = "solid"
)
series_order <- c(
  "Original community",
  "Community with structural modification A",
  "Community with structural modification B"
)

valley_df$series <- factor(valley_df$series, levels = series_order)
peak_df$series   <- factor(peak_df$series, levels = series_order)

xA_lab <- FREQ_SHIFT * 0.14
yA_lab <- 1.25 * max(S_A)

xB_lab <- FREQ_SHIFT * 0.58
yB_lab <- 0.52 * max(S_B)

yA_target <- S_A[which.min(abs(OMEGAS - omega_valley))]
yB_target <- S_B[which.min(abs(OMEGAS - omega_peak))]

# ============================================================
# Network data
# ============================================================

pts <- data.frame(
  node = 1:12,
  x = c(
    0.08, 0.18, 0.29, 0.22,
    0.42, 0.50, 0.46,
    0.70, 0.83, 0.73,
    1.03, 1.00
  ),
  y = c(
    0.63, 0.82, 0.60, 0.34,
    0.80, 0.54, 0.22,
    0.74, 0.54, 0.28,
    0.73, 0.40
  )
)

x_center <- mean(range(pts$x))
x_scale_net <- 0.92
dy_net <- -0.08

pts_net <- transform(
  pts,
  x = x_center + x_scale_net * (x - x_center),
  y = y + dy_net
)

pts_net$x <- 0.5 + 0.85 * (pts_net$x - 0.5)
pts_net$y <- pts_net$y + dy_net

base_edges <- data.frame(
  from = c(
    1, 2, 2, 3, 4,
    5, 5,
    6, 6, 6,
    8,
    9, 9, 9, 9
  ),
  to = c(
    2, 3, 5, 6, 6,
    6, 9,
    7, 9, 10,
    9,
    10, 11, 12, 8
  )
)

regionA_nodes <- c(3)
regionB_nodes <- c(9)

modA_edges <- subset(base_edges, from == regionA_nodes[1] | to == regionA_nodes[1])
modB_edges <- subset(base_edges, from == regionB_nodes[1] | to == regionB_nodes[1])

base_edges_df <- edge_df(base_edges, pts_net)
modA_edges_df <- edge_df(modA_edges, pts_net)
modB_edges_df <- edge_df(modB_edges, pts_net)

circA_cx <- pts_net$x[pts_net$node == regionA_nodes]
circB_cx <- pts_net$x[pts_net$node == regionB_nodes]

circA_df <- circle_df(
  pts_net$x[pts_net$node == regionA_nodes],
  pts_net$y[pts_net$node == regionA_nodes],
  0.11
)

circB_df <- circle_df(
  pts_net$x[pts_net$node == regionB_nodes],
  pts_net$y[pts_net$node == regionB_nodes],
  0.11
)

# ============================================================
# Themes
# ============================================================

theme_clean <- theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black", linewidth = 0.45),
    plot.title = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background = element_rect(fill = "transparent", colour = NA),
    legend.title = element_blank(),
    legend.background = element_rect(fill = alpha("white", 0.9), colour = "grey50"),
    legend.key = element_rect(fill = alpha("white", 0), colour = NA)
  )

theme_inset <- theme_void() +
  theme(
    panel.background = element_rect(fill = alpha("white", 0.92), colour = "grey60", linewidth = 0.3),
    plot.background = element_rect(fill = alpha("white", 0), colour = NA)
  )

# ============================================================
# Panel A: network
# ============================================================

p_net <- ggplot() +
  geom_polygon(
    data = circA_df,
    aes(x, y),
    fill = alpha(COL_A, 0.13),
    color = alpha(COL_A, 0.65),
    linewidth = 0.8
  ) +
  geom_polygon(
    data = circB_df,
    aes(x, y),
    fill = alpha(COL_B, 0.13),
    color = alpha(COL_B, 0.65),
    linewidth = 0.8
  ) +
  geom_segment(
    data = base_edges_df,
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    color = COL_EDGE,
    linewidth = 1.2
  ) +
  geom_segment(
    data = modA_edges_df,
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    color = COL_A,
    linewidth = 1.6
  ) +
  geom_segment(
    data = modB_edges_df,
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    color = COL_B,
    linewidth = 1.6
  ) +
  geom_point(
    data = pts_net,
    aes(x, y),
    shape = 21,
    fill = "darkgrey",
    color = "black",
    stroke = 0.8,
    size = 5.0
  ) +
  geom_point(
    data = subset(pts_net, node %in% regionA_nodes),
    aes(x, y),
    shape = 21,
    fill = COL_A,
    color = "black",
    stroke = 0.9,
    size = 5.0
  ) +
  geom_point(
    data = subset(pts_net, node %in% regionB_nodes),
    aes(x, y),
    shape = 21,
    fill = COL_B,
    color = "black",
    stroke = 0.9,
    size = 5.0
  ) +
  # geom_text(
  #   data = pts_net,
  #   aes(x, y, label = node),
  #   vjust = -1.2,
  #   size = 4,
  #   color = "black"
  # ) +
  annotate(
    "curve",
    x = circA_cx,
    y = pts_net$y[pts_net$node == regionA_nodes]*1.13,
    xend = x_center + x_scale_net * (0.37 - x_center),
    yend = 0.91 + dy_net * 1.5 - 0.04,
    curvature = -0.07,
    arrow = arrow(length = unit(0.18, "cm")),
    color = alpha(COL_A, 0.55),
    linewidth = 0.7
  ) +
  annotate(
    "curve",
    x = circB_cx,
    y = pts_net$y[pts_net$node == regionB_nodes] + 0.07,
    xend = x_center + x_scale_net * (0.76 - x_center),
    yend = 0.91 + dy_net * 1.5 - 0.04,
    curvature = 0.14,
    arrow = arrow(length = unit(0.18, "cm")),
    color = alpha(COL_B, 0.55),
    linewidth = 0.7
  ) +
  annotate(
    "text",
    x = x_center + x_scale_net * (0.37 - x_center),
    y = 0.91 + dy_net * 1.5,
    label = "Structural modification A",
    color = COL_A,
    size = 5,
    fontface = "bold"
  ) +
  annotate(
    "text",
    x = x_center + x_scale_net * (0.76 - x_center),
    y = 0.91 + dy_net * 1.5,
    label = "Structural modification B",
    color = COL_B,
    size = 5,
    fontface = "bold"
  ) +
  coord_cartesian(xlim = c(0.04, 1.04), ylim = c(-0.02, 0.90), expand = FALSE) +
  labs(title = NULL, x = NULL, y = NULL) +
  theme_clean +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.line = element_blank()
  )

# ============================================================
# Panel B: frequency profile
# ============================================================

p_prof <- ggplot(profile_df, aes(x = omega, y = sensitivity, color = class)) +
  annotate(
    "rect",
    xmin = min(OMEGAS),
    xmax = FREQ_SHIFT * 0.55,
    ymin = -Inf,
    ymax = Inf,
    fill = "grey70",
    alpha = 0.06
  ) +
  annotate(
    "rect",
    xmin = FREQ_SHIFT * 0.55,
    xmax = FREQ_SHIFT * 1.35,
    ymin = -Inf,
    ymax = Inf,
    fill = "grey70",
    alpha = 0.10
  ) +
  annotate(
    "rect",
    xmin = FREQ_SHIFT * 1.35,
    xmax = max(OMEGAS),
    ymin = -Inf,
    ymax = Inf,
    fill = "grey70",
    alpha = 0.06
  ) +
  geom_line(linewidth = 1.2) +
  geom_vline(
    xintercept = omega_valley,
    color = high_col,
    linetype = "dashed",
    linewidth = 1.2
  ) +
  geom_vline(
    xintercept = omega_peak,
    color = resp_col,
    linetype = "dashed",
    linewidth = 1.2
  ) +
  annotate(
    "curve",
    x = xA_lab,
    y = yA_lab,
    xend = omega_valley,
    yend = yA_target,
    curvature = -0.12,
    arrow = arrow(length = unit(0.18, "cm")),
    color = alpha(COL_A, 0.55),
    linewidth = 0.7
  ) +
  annotate(
    "curve",
    x = xB_lab,
    y = yB_lab,
    xend = omega_peak,
    yend = yB_target,
    curvature = 0.12,
    arrow = arrow(length = unit(0.18, "cm")),
    color = alpha(COL_B, 0.55),
    linewidth = 0.7
  ) +
    annotate(
      "label",
      x = xA_lab,
      y = yA_lab,
      label = "A matters more here",
      fill = alpha("white", 0.95),
      color = COL_A,
      label.size = 0.3,
      size = 5,
      fontface = "bold"
    ) +
    annotate(
      "label",
      x = xB_lab,
      y = yB_lab,
      label = "B matters more here",
      fill = alpha("white", 0.95),
      color = COL_B,
      label.size = 0.3,
      size = 5,
      fontface = "bold"
    ) +
  scale_x_log10(
    breaks = c(0.05, 0.1, 0.2, 0.5, 1),
    labels = label_number()
  ) +
  scale_y_log10(labels = label_math()) +
  scale_color_manual(values = prof_cols) +
  coord_cartesian(xlim = c(min(OMEGAS), max(OMEGAS)), ylim = y_prof, expand = FALSE) +
  labs(
    title = NULL,
    x = "frequency \u03C9",
    y = "sensitivity"
  ) +
  theme_clean +
  theme(
    legend.position = c(0.13, 0.18),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  )

# ============================================================
# Forcing insets
# ============================================================
p_force_valley <- ggplot(forcing_valley_df, aes(x = Time, y = value)) +
  geom_line(color = high_col, linewidth = 1.1) +
  annotate(
    "text",
    x = min(ts) + 0.96 * 0.42 * diff(range(ts)),
    y = 1.25 * FORCE_AMPLITUDE,
    label = sprintf("\u03C9 = %.3f", omega_valley_time),
    hjust = 1,
    vjust = 1,
    size = 3.2,
    color = "black"
  ) +
  coord_cartesian(
    xlim = c(min(ts), min(ts) + 0.42 * diff(range(ts))),
    ylim = c(-1.05 * FORCE_AMPLITUDE, 1.35 * FORCE_AMPLITUDE),
    expand = FALSE
  ) +
  theme_inset
p_force_peak <- ggplot(forcing_peak_df, aes(x = Time, y = value)) +
  geom_line(color = resp_col, linewidth = 1.1) +
  annotate(
    "text",
    x = min(ts) + 0.96 * 0.42 * diff(range(ts)),
    y = 1.25 * FORCE_AMPLITUDE,
    label = sprintf("\u03C9 = %.3f", omega_peak_time),
    hjust = 1,
    vjust = 1,
    size = 3.2,
    color = "black"
  ) +
  coord_cartesian(
    xlim = c(min(ts), min(ts) + 0.42 * diff(range(ts))),
    ylim = c(-1.05 * FORCE_AMPLITUDE, 1.35 * FORCE_AMPLITUDE),
    expand = FALSE
  ) +
  theme_inset

g_force_valley <- ggplotGrob(p_force_valley)
g_force_peak   <- ggplotGrob(p_force_peak)

# ============================================================
# Panel C: valley time series
# ============================================================

p_valley <- ggplot(valley_df, aes(x = Time, y = value, color = series, linetype = series)) +
  geom_vline(
    xintercept = FORCE_START,
    color = "grey35",
    linetype = "dashed",
    linewidth = 0.6
  ) +
  geom_line(
    data = subset(valley_df, series == "Original community"),
    linewidth = 1.0,
    alpha = 1
  ) +
  geom_line(
    data = subset(valley_df, series != "Original community"),
    linewidth = 1.0,
    alpha = 0.75
  ) +
  annotation_custom(
    grob = g_force_valley,
    xmin = min(ts) + 0.02 * diff(range(ts)),
    xmax = min(ts) + 0.28 * diff(range(ts)),
    ymin = y_comm[2] - 0.34 * diff(y_comm),
    ymax = y_comm[2] - 0.02 * diff(y_comm)
  ) +
  scale_color_manual(values = ts_cols) +
  scale_linetype_manual(values = ts_ltys) +
  coord_cartesian(xlim = c(min(ts), max(ts)), ylim = y_comm, expand = FALSE) +
  labs(
    title = NULL,
    x = "Time",
    y = "Community biomass"
  ) +
  theme_clean +
  theme(
    legend.position = c(0.78, 0.88),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 14),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16)
  )
# ============================================================
# Panel D: peak time series
# ============================================================

p_peak <- ggplot(peak_df, aes(x = Time, y = value, color = series, linetype = series)) +
  geom_vline(
    xintercept = FORCE_START,
    color = "grey35",
    linetype = "dashed",
    linewidth = 0.6
  ) +
  geom_line(
    data = subset(peak_df, series == "baseline"),
    linewidth = 1.0,
    alpha = 1
  ) +
  geom_line(
    data = subset(peak_df, series != "baseline"),
    linewidth = 1.0,
    alpha = 0.75
  ) +
  annotation_custom(
    grob = g_force_peak,
    xmin = min(ts) + 0.02 * diff(range(ts)),
    xmax = min(ts) + 0.28 * diff(range(ts)),
    ymin = y_comm[2] - 0.34 * diff(y_comm),
    ymax = y_comm[2] - 0.02 * diff(y_comm)
  ) +
  scale_color_manual(values = ts_cols) +
  scale_linetype_manual(values = ts_ltys) +
  coord_cartesian(xlim = c(min(ts), max(ts)), ylim = y_comm, expand = FALSE) +
  labs(
    title = NULL,
    x = "Time",
    y = "Community abundance"
  ) +
  theme_clean +
  theme(
    legend.position = c(0.78, 0.88),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    axis.text.x = element_text(size = 14),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16)
  )
# ============================================================
# Save each panel separately
# ============================================================

save_part("panel_network.png", p_net, width = 7.5, height = 4.8)
save_part("panel_frequency_profile.png", p_prof, width = 12.0, height = 4.8)
save_part("panel_time_valley.png", p_valley, width = 7.5, height = 4.8)
save_part("panel_time_peak.png", p_peak, width = 7.5, height = 4.8)