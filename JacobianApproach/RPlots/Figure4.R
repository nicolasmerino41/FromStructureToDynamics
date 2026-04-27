# ============================================================
# Independent R script: multifrequency forcing figure
# ============================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(cowplot)
library(magick)
library(grid)


# -------------------------
# Parameters
# -------------------------

omega_low  <- 0.08
omega_mid  <- 0.62
omega_high <- 2.8

d_low  <- 5.0
d_mid  <- 5e-5
d_high <- 5.0

eps12 <- 1e-8
eps23 <- 1e-8

dt <- 0.02
tmax <- 600
burn_time <- 300

# -------------------------
# Matrix A
# -------------------------

A <- matrix(c(
  -d_low,      0,           eps12,       0,           0,          0,
  0,         -d_low,       0,           eps12,       0,          0,
  
  eps12,      0,          -d_mid,      -omega_mid,  eps23,      0,
  0,          eps12,       omega_mid,  -d_mid,      0,          eps23,
  
  0,          0,           eps23,       0,          -d_high,    0,
  0,          0,           0,           eps23,       0,         -d_high
), nrow = 6, byrow = TRUE)

# -------------------------
# Resolvent profile
# -------------------------

resolvent_norm <- function(A, omega) {
  n <- nrow(A)
  M <- 1i * omega * diag(n) - A
  R <- solve(M)
  svd(R, nu = 0, nv = 0)$d[1]
}

omega_s <- exp(seq(log(0.03), log(8), length.out = 12000))
S <- vapply(omega_s, function(w) resolvent_norm(A, w), numeric(1))

df_profile <- tibble(
  omega = omega_s,
  S = S
)

# -------------------------
# Forcing
# -------------------------

b <- c(0.35, -0.25, 1.00, -0.85, 0.30, -0.20)
b <- b / sqrt(sum(b^2))

a_low  <- 0.12
a_mid  <- 0.12
a_high <- 0.12

phi_low  <- 0.4
phi_mid  <- 1.1
phi_high <- -0.7

low_component <- function(t) {
  a_low * sin(omega_low * t + phi_low)
}

mid_component <- function(t) {
  a_mid * sin(omega_mid * t + phi_mid)
}

high_component <- function(t) {
  a_high * sin(omega_high * t + phi_high)
}

forcing_scalar <- function(t) {
  low_component(t) + mid_component(t) + high_component(t)
}

forcing_fun <- function(t) {
  forcing_scalar(t) * b
}

# -------------------------
# Icon combination: icon + icon + icon
# -------------------------
# This version uses draw_plot() for the border and draw_image() for the PNG.
# It avoids grid coordinate mismatch from draw_grob().

if (
  file.exists("RPlots/perturbation_1.png") &&
  file.exists("RPlots/perturbation_2.png") &&
  file.exists("RPlots/perturbation_3.png")
) {
  icon_low  <- image_read("RPlots/perturbation_1.png")
  icon_mid  <- image_read("RPlots/perturbation_2.png")
  icon_high <- image_read("RPlots/perturbation_3.png")
  
  icon_border <- ggplot() +
    annotate(
      "rect",
      xmin = 0, xmax = 1,
      ymin = 0, ymax = 1,
      fill = NA,
      color = "black",
      linewidth = 2.5
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    theme_void()
  
  add_icon <- function(fig_obj, img, x, y, w = 0.105, h = 0.105) {
    fig_obj +
      draw_image(img, x = x, y = y, width = w, height = h) +
      draw_plot(icon_border, x = x, y = y, width = w, height = h)
  }
  
  fig <- add_icon(fig, icon_low,  x = 0.405, y = 0.770)
  fig <- add_icon(fig, icon_mid,  x = 0.405, y = 0.655)
  fig <- add_icon(fig, icon_high, x = 0.405, y = 0.540)
}
# -------------------------
# RK4 simulation
# -------------------------

simulate_system <- function(A, forcing_fun, dt, tmax) {
  ts <- seq(0, tmax, by = dt)
  n <- nrow(A)
  X <- matrix(0, nrow = length(ts), ncol = n)
  
  f <- function(x, t) {
    as.numeric(A %*% x + forcing_fun(t))
  }
  
  x <- rep(0, n)
  
  for (k in seq_len(length(ts) - 1)) {
    t <- ts[k]
    
    k1 <- f(x, t)
    k2 <- f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 <- f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 <- f(x + dt * k3, t + dt)
    
    x <- x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    X[k + 1, ] <- x
  }
  
  list(time = ts, X = X)
}

sim <- simulate_system(A, forcing_fun, dt, tmax)

ts <- sim$time
X <- sim$X

c_readout <- c(0, 0, 1, 1, 0, 0)
response <- as.numeric(X %*% c_readout)

df_time <- tibble(
  time = ts,
  low = low_component(ts),
  mid = mid_component(ts),
  high = high_component(ts),
  forcing = forcing_scalar(ts),
  response = response
) |>
  filter(time >= burn_time)

df_components <- df_time |>
  select(time, low, mid, high) |>
  pivot_longer(
    cols = c(low, mid, high),
    names_to = "component",
    values_to = "value"
  ) |>
  mutate(
    component = factor(
      component,
      levels = c("low", "mid", "high"),
      labels = c("slow", "intermediate", "fast")
    )
  )
# -------------------------
# Styling
# -------------------------

low_col  <- "#2C7FB8"
mid_col  <- "#D95F02"
high_col <- "#1B9E77"
resp_col <- "#7B3294"
flow_col <- "#F2F6EA"

force_scale <- max(abs(df_time$forcing), na.rm = TRUE)
resp_scale  <- max(abs(df_time$response), na.rm = TRUE)

df_time <- df_time |>
  mutate(response_scaled = response / resp_scale * force_scale)

# Display-only profile lines
profile_low_line  <- 0.09
profile_mid_line  <- 0.62
profile_high_line <- 3.8

# -------------------------
# Profile plot, with large external box
# -------------------------
p_profile_inner <- ggplot(df_profile, aes(omega, S)) +
  geom_line(linewidth = 0.9, color = "black") +
  geom_vline(xintercept = profile_low_line,
             linetype = "dashed", color = low_col, linewidth = 1.05) +
  geom_vline(xintercept = profile_mid_line,
             linetype = "dashed", color = mid_col, linewidth = 1.05) +
  geom_vline(xintercept = profile_high_line,
             linetype = "dashed", color = high_col, linewidth = 1.05) +
  scale_x_log10(limits = c(0.03, 8), breaks = NULL) +
  scale_y_log10() +
  labs(x = expression(omega), y = "sensitivity") +
  theme_classic(base_size = 10) +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.line = element_line(linewidth = 0.35),
    axis.title.x = element_text(size = 9, margin = margin(t = 1)),
    axis.title.y = element_text(size = 9, margin = margin(r = 1)),
    plot.margin = margin(0, 0, 0, 0)
  )
# -------------------------
# Small neutral network inset for profile panel
# -------------------------

COL_EDGE <- "#ABABAB"
COL_NODE <- "#2E2E2E"

pts <- data.frame(
  node = 1:12,
  x = c(0.08, 0.18, 0.29, 0.22,
        0.42, 0.50, 0.46,
        0.70, 0.83, 0.73,
        1.03, 1.00),
  y = c(0.63, 0.82, 0.60, 0.34,
        0.80, 0.54, 0.22,
        0.74, 0.54, 0.28,
        0.73, 0.40)
)

x_center <- mean(range(pts$x))

pts_net <- transform(
  pts,
  x = 0.5 + 0.78 * (x - x_center),
  y = y - 0.08
)

base_edges <- data.frame(
  from = c(1,2,2,3,4,5,5,6,6,6,8,9,9,9,9),
  to   = c(2,3,5,6,6,6,9,7,9,10,9,10,11,12,8)
)

edge_df <- function(edges, pts_df) {
  merge(edges, pts_df, by.x = "from", by.y = "node") |>
    merge(pts_df, by.x = "to", by.y = "node", suffixes = c("_from", "_to"))
}

base_edges_df <- edge_df(base_edges, pts_net)

p_net_inset <- ggplot() +
  geom_segment(
    data = base_edges_df,
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    color = COL_EDGE,
    linewidth = 0.55
  ) +
  geom_point(
    data = pts_net,
    aes(x, y),
    color = COL_NODE,
    size = 1.9
  ) +
  coord_cartesian(xlim = c(0.02, 1.04), ylim = c(0.02, 0.86), expand = FALSE) +
  theme_void() +
  theme(
    plot.background = element_rect(fill = NA, color = NA),
    panel.background = element_rect(fill = NA, color = NA)
  )
p_net_inset_rotated <- editGrob(
  ggplotGrob(p_net_inset),
  vp = viewport(angle = 24)
)

p_profile_box <- ggdraw() +
  draw_plot(
    p_profile_inner,
    x = 0,
    y = 0,
    width = 1,
    height = 1
  ) +
  draw_grob(
    p_net_inset_rotated,
    x = 0.035,
    y = 0.705,
    width = 0.22,
    height = 0.20
  )
# -------------------------
# Three single-frequency inputs
# -------------------------

p_components <- ggplot(df_components, aes(time, value, color = component)) +
  geom_line(linewidth = 0.85) +
  facet_wrap(~component, ncol = 1) +
  scale_color_manual(
    values = c(
      "slow" = low_col,
      "intermediate" = mid_col,
      "fast" = high_col
    )
  ) +
  coord_cartesian(xlim = c(300, 365)) +
  theme_void(base_size = 11) +
  theme(
    legend.position = "none",
    strip.text = element_blank(),
    panel.spacing = unit(1.15, "lines"),
    plot.margin = margin(12, 24, 12, 16)
  )
# -------------------------
# Compound perturbation
# -------------------------

p_forcing <- ggplot(df_time, aes(time, forcing)) +
  geom_line(linewidth = 1.05, color = "black") +
  coord_cartesian(
    xlim = c(300, 365),
    ylim = c(-1.25 * force_scale, 1.25 * force_scale)
  ) +
  theme_void(base_size = 11) +
  theme(plot.margin = margin(12, 22, 10, 22))

# -------------------------
# Community response
# -------------------------

p_response <- ggplot(df_time, aes(time)) +
  geom_line(
    aes(y = forcing, color = "multifrequency forcing"),
    linewidth = 0.45
  ) +
  geom_line(
    aes(y = response_scaled, color = "community biomass"),
    linewidth = 1.25
  ) +
  scale_color_manual(
    values = c(
      "multifrequency forcing" = "grey72",
      "community biomass" = resp_col
    )
  ) +
  coord_cartesian(
    xlim = c(300, 365),
    ylim = c(-1.35 * force_scale, 1.35 * force_scale)
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position = c(0.21, 0.86),
    legend.title = element_blank(),
    legend.text = element_text(size = 8.5),
    legend.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(8, 22, 12, 22)
  )

# -------------------------
# Main layout
# Components above multi-response, profile centered top-right
# -------------------------
# -------------------------
# New layout
# top-left    = unifrequencies
# top-right   = multifrequency
# bottom-left = profile
# bottom-right= community response
# -------------------------

main_fig <- (
  (p_components | p_forcing) /
    (p_profile_box | p_response)
) +
  plot_layout(
    widths = c(1.10, 1.10),
    heights = c(1.22, 1.00)
  ) &
  theme(plot.background = element_rect(fill = "white", color = NA))

# -------------------------
# Background + three arrows into multi
# -------------------------
fig <- ggdraw() +
  draw_grob(
    rectGrob(
      x = 0.50,
      y = 0.50,
      width = 0.94,
      height = 0.82,
      gp = gpar(fill = flow_col, col = NA, alpha = 0.95)
    )
  ) +
  draw_plot(main_fig) +
  
  # three shorter, wider arrows converging to same spot on multi plot
  draw_grob(
    curveGrob(
      x1 = 0.475, y1 = 0.755,
      x2 = 0.535, y2 = 0.645,
      curvature = -0.18,
      arrow = arrow(length = unit(0.30, "cm"), type = "closed"),
      gp = gpar(linewidth = 1.65)
    )
  ) +
  draw_grob(
    curveGrob(
      x1 = 0.475, y1 = 0.645,
      x2 = 0.535, y2 = 0.645,
      curvature = 0,
      arrow = arrow(length = unit(0.30, "cm"), type = "closed"),
      gp = gpar(linewidth = 1.65)
    )
  ) +
  draw_grob(
    curveGrob(
      x1 = 0.475, y1 = 0.535,
      x2 = 0.535, y2 = 0.645,
      curvature = 0.18,
      arrow = arrow(length = unit(0.30, "cm"), type = "closed"),
      gp = gpar(linewidth = 1.65)
    )
  )

# place combined icon on top-right corner of multifrequency plot
if (!is.null(combo_icon)) {
  fig <- fig +
    draw_plot(
      combo_icon,
      x = 0.73,
      y = 0.89,
      width = 0.25,
      height = 0.105
    )
}
fig

# -------------------------
# Optional perturbation icons
# Positioned over the top-right corner of each unifrequency panel
# -------------------------
if (
  file.exists("RPlots/perturbation_1.png") &&
  file.exists("RPlots/perturbation_2.png") &&
  file.exists("RPlots/perturbation_3.png")
) {
  icon_low  <- image_read("RPlots/perturbation_1.png")
  icon_mid  <- image_read("RPlots/perturbation_2.png")
  icon_high <- image_read("RPlots/perturbation_3.png")
  
  fig <- ggdraw(fig) +
    draw_image(icon_low,  x = 0.02, y = 0.87, width = 0.12, height = 0.12) +
    draw_image(icon_mid,  x = 0.02, y = 0.7, width = 0.13, height = 0.13) +
    draw_image(icon_high, x = 0.02, y = 0.53, width = 0.13, height = 0.13)
}

fig

# -------------------------
# Save
# -------------------------

ggsave(
  "RPlots/multifrequency_resolvent_figure.png",
  fig,
  width = 10.5,
  height = 6.7,
  dpi = 450
)

fig