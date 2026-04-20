# ============================================================
# four_architectures_same_resilience.R
#
# Single R script combining the three Julia scripts:
#   Row 1: network representations
#   Row 2: matrix heatmaps
#   Row 3: intrinsic sensitivity profiles
#
# Four systems with the same asymptotic resilience
# (same spectral abscissa alpha(A) = target_alpha):
#   1) symmetric normal
#   2) non-symmetric normal
#   3) triangular
#   4) modular
#
# Extra:
# - matrix "titles" are drawn as colored label boxes
# - label colors match the bottom profile line colors
# - those colors are sampled from the same balance-style
#   diverging palette used for the heatmaps
# ============================================================

# -------------------------
# Packages
# -------------------------
library(ggplot2)
library(patchwork)
library(scales)
library(grid)

# ============================================================
# Helpers
# ============================================================

opnorm2 <- function(M) {
  max(svd(M, nu = 0, nv = 0)$d)
}

resolvent <- function(A, Tmat, omega) {
  n <- nrow(A)
  solve(1i * omega * Tmat - A)
}

intrinsic_sensitivity <- function(A, Tmat, omega) {
  opnorm2(resolvent(A, Tmat, omega))
}

structured_sensitivity <- function(A, Tmat, P, omega) {
  R <- resolvent(A, Tmat, omega)
  opnorm2(R %*% P %*% R)
}

profile_intrinsic <- function(A, Tmat, omegas) {
  vapply(omegas, function(w) intrinsic_sensitivity(A, Tmat, w), numeric(1))
}

profile_structured <- function(A, Tmat, P, omegas) {
  vapply(omegas, function(w) structured_sensitivity(A, Tmat, P, w), numeric(1))
}

trapz <- function(x, y) {
  sum(0.5 * (y[-length(y)] + y[-1]) * diff(x))
}

spectral_abscissa <- function(A) {
  max(Re(eigen(A, only.values = TRUE)$values))
}

resilience <- function(A) {
  -spectral_abscissa(A)
}

enforce_alpha <- function(A, target_alpha) {
  alpha_now <- spectral_abscissa(A)
  A + (target_alpha - alpha_now) * diag(nrow(A))
}

orthogonal_matrix <- function(n, seed = 1) {
  set.seed(seed)
  Q <- qr.Q(qr(matrix(rnorm(n * n), nrow = n)))
  unclass(Q)
}

interaction_matrix <- function(A) {
  A - diag(diag(A))
}

match_offdiag_frobenius <- function(A, target_frob) {
  D <- diag(diag(A))
  G <- A - D
  ng <- norm(G, type = "F")
  if (ng == 0) return(A)
  D + (target_frob / ng) * G
}

interaction_stats <- function(A) {
  G <- interaction_matrix(A)
  total_abs <- sum(abs(G))
  frob <- norm(G, type = "F")
  op2 <- opnorm2(G)
  maxrow <- max(rowSums(abs(G)))
  c(total_abs = total_abs, frob = frob, op2 = op2, maxrow = maxrow)
}

# ============================================================
# Matrix builders
# ============================================================

build_symmetric_normal <- function(lambda, seed = 10) {
  n <- length(lambda)
  Q <- orthogonal_matrix(n, seed = seed)
  A <- Q %*% diag(lambda) %*% t(Q)
  0.5 * (A + t(A))
}

build_nonsymmetric_normal <- function(alphas, betas, seed = 20) {
  stopifnot(length(alphas) == length(betas))
  k <- length(alphas)
  n <- 2 * k
  B <- matrix(0, n, n)
  for (i in seq_len(k)) {
    a <- alphas[i]
    b <- betas[i]
    idx <- (2 * i - 1):(2 * i)
    B[idx, idx] <- matrix(c(-a, -b, b, -a), 2, 2, byrow = TRUE)
  }
  Q <- orthogonal_matrix(n, seed = seed)
  Q %*% B %*% t(Q)
}

build_triangular_oscillatory <- function() {
  B1 <- matrix(c(-0.35, -0.9,  0.9, -0.35), 2, 2, byrow = TRUE)
  B2 <- matrix(c(-0.55, -1.7,  1.7, -0.55), 2, 2, byrow = TRUE)
  B3 <- matrix(c(-0.85, -2.8,  2.8, -0.85), 2, 2, byrow = TRUE)
  B4 <- matrix(c(-1.20, -4.2,  4.2, -1.20), 2, 2, byrow = TRUE)
  
  A <- matrix(0, 8, 8)
  A[1:2, 1:2] <- B1
  A[3:4, 3:4] <- B2
  A[5:6, 5:6] <- B3
  A[7:8, 7:8] <- B4
  
  A[1:2, 3:4] <- matrix(c(1.2, 0.3, -0.4, 0.9), 2, 2, byrow = TRUE)
  A[3:4, 5:6] <- matrix(c(1.0, -0.2, 0.5, 0.8), 2, 2, byrow = TRUE)
  A[5:6, 7:8] <- matrix(c(0.9, 0.4, -0.3, 0.7), 2, 2, byrow = TRUE)
  
  A[1:2, 5:6] <- matrix(c(0.45, 0.10, -0.15, 0.35), 2, 2, byrow = TRUE)
  A[3:4, 7:8] <- matrix(c(0.35, -0.08, 0.12, 0.28), 2, 2, byrow = TRUE)
  A[1:2, 7:8] <- matrix(c(0.18, 0.00, 0.00, 0.14), 2, 2, byrow = TRUE)
  
  A
}

build_modular_oscillatory <- function() {
  A <- matrix(0, 8, 8)
  
  C1a <- matrix(c(-0.35, -0.8,  0.8, -0.35), 2, 2, byrow = TRUE)
  C1b <- matrix(c(-0.55, -1.4,  1.4, -0.55), 2, 2, byrow = TRUE)
  C2a <- matrix(c(-0.40, -2.2,  2.2, -0.40), 2, 2, byrow = TRUE)
  C2b <- matrix(c(-0.65, -3.6,  3.6, -0.65), 2, 2, byrow = TRUE)
  
  C1 <- matrix(0, 4, 4)
  C1[1:2, 1:2] <- C1a
  C1[3:4, 3:4] <- C1b
  C1[1:2, 3:4] <- matrix(c(0.45, 0.12, -0.08, 0.35), 2, 2, byrow = TRUE)
  C1[3:4, 1:2] <- matrix(c(0.20, 0.00, 0.00, 0.18), 2, 2, byrow = TRUE)
  
  C2 <- matrix(0, 4, 4)
  C2[1:2, 1:2] <- C2a
  C2[3:4, 3:4] <- C2b
  C2[1:2, 3:4] <- matrix(c(0.38, -0.06, 0.10, 0.30), 2, 2, byrow = TRUE)
  C2[3:4, 1:2] <- matrix(c(0.16, 0.00, 0.00, 0.14), 2, 2, byrow = TRUE)
  
  A[1:4, 1:4] <- C1
  A[5:8, 5:8] <- C2
  
  A[1:4, 5:8] <- matrix(c(
    0.00, 0.05, 0.00, 0.02,
    0.03, 0.00, 0.04, 0.00,
    0.00, 0.04, 0.00, 0.03,
    0.02, 0.00, 0.03, 0.00
  ), 4, 4, byrow = TRUE)
  
  A[5:8, 1:4] <- matrix(c(
    0.00, 0.02, 0.00, 0.01,
    0.01, 0.00, 0.02, 0.00,
    0.00, 0.02, 0.00, 0.01,
    0.01, 0.00, 0.01, 0.00
  ), 4, 4, byrow = TRUE)
  A
}

community_perturbation <- function(n1, n2, cross_weight = 0.35) {
  n <- n1 + n2
  P <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i != j) {
        same_comm <- (i <= n1 && j <= n1) || (i > n1 && j > n1)
        P[i, j] <- if (same_comm) 1.0 else cross_weight
      }
    }
  }
  P
}

# ============================================================
# Shared frequency grid and perturbation class
# ============================================================

omegas <- 10 ^ seq(-1.5, 1.1, length.out = 900)
T_hom <- diag(8)
P <- community_perturbation(4, 4, cross_weight = 0.35)

# ============================================================
# Build four cases with the same resilience
# ============================================================

target_alpha <- -0.35
target_resilience <- -target_alpha
target_frob <- 3.5

lambda_sym <- c(-0.35, -0.55, -0.80, -1.05, -1.30, -1.55, -1.80, -2.10)
A_sym <- build_symmetric_normal(lambda_sym, seed = 11)
A_sym <- enforce_alpha(A_sym, target_alpha)

alphas_ns <- c(0.35, 0.70, 1.20, 1.75)
betas_ns  <- c(0.80, 1.45, 2.60, 4.10)
A_nsn <- build_nonsymmetric_normal(alphas_ns, betas_ns, seed = 22)
A_nsn <- enforce_alpha(A_nsn, target_alpha)

A_tri <- build_triangular_oscillatory()
A_tri <- enforce_alpha(A_tri, target_alpha)

A_mod <- build_modular_oscillatory()
A_mod <- enforce_alpha(A_mod, target_alpha)

A_sym <- match_offdiag_frobenius(A_sym, target_frob)
A_nsn <- match_offdiag_frobenius(A_nsn, target_frob)
A_tri <- match_offdiag_frobenius(A_tri, target_frob)
A_mod <- match_offdiag_frobenius(A_mod, target_frob)

A_sym <- enforce_alpha(A_sym, target_alpha)
A_nsn <- enforce_alpha(A_nsn, target_alpha)
A_tri <- enforce_alpha(A_tri, target_alpha)
A_mod <- enforce_alpha(A_mod, target_alpha)

cases <- list(
  list(name = "Symmetric normal",     A = A_sym),
  list(name = "Non-symmetric normal", A = A_nsn),
  list(name = "Triangular",           A = A_tri),
  list(name = "Modular",              A = A_mod)
)

# ============================================================
# Sensitivity profiles
# ============================================================

intr_profiles <- list()
struct_profiles <- list()

for (case in cases) {
  intr_profiles[[case$name]]   <- profile_intrinsic(case$A, T_hom, omegas)
  struct_profiles[[case$name]] <- profile_structured(case$A, T_hom, P, omegas)
}

# ============================================================
# Diagnostics
# ============================================================

cat("============================================================\n")
cat("Four cases with the same resilience\n")
cat("============================================================\n")
cat(sprintf("Target spectral abscissa alpha* = %.4f\n", target_alpha))
cat(sprintf("Target resilience        r*     = %.4f\n\n", target_resilience))

for (case in cases) {
  A <- case$A
  alpha_now <- spectral_abscissa(A)
  r_now <- resilience(A)
  normal_defect <- opnorm2(t(A) %*% A - A %*% t(A))
  cat(sprintf(
    "%-24s  alpha(A) = % .6f   resilience = %.6f   normal defect = %.3e\n",
    case$name, alpha_now, r_now, normal_defect
  ))
}

cat("\n============================================================\n")
cat("Peak sensitivities\n")
cat("============================================================\n")
for (case in cases) {
  yi <- intr_profiles[[case$name]]
  ys <- struct_profiles[[case$name]]
  ii <- which.max(yi)
  jj <- which.max(ys)
  cat(sprintf("%-24s intrinsic max  = %10.4f at omega = %.4g\n", case$name, yi[ii], omegas[ii]))
  cat(sprintf("%-24s structured max = %10.4f at omega = %.4g\n\n", "", ys[jj], omegas[jj]))
}

cat("Integrated sensitivities\n")
for (case in cases) {
  ai <- trapz(omegas, intr_profiles[[case$name]])
  as <- trapz(omegas, struct_profiles[[case$name]])
  cat(sprintf("%-24s integral intrinsic = %10.4f   integral structured = %10.4f\n",
              case$name, ai, as))
}

cat("\n============================================================\n")
cat("Interaction-strength diagnostics\n")
cat("============================================================\n")
for (case in cases) {
  st <- interaction_stats(case$A)
  cat(sprintf(
    "%-24s total|offdiag| = %8.4f   ||offdiag||F = %8.4f   ||offdiag||2 = %8.4f   max row sum = %8.4f\n",
    case$name, st["total_abs"], st["frob"], st["op2"], st["maxrow"]
  ))
}

# ============================================================
# Color system
# ============================================================

balance_palette <- colorRampPalette(c(
  "#1E2F97",  # darker blue extreme
  "#5B6FD6",
  "#F3F3F3",
  "#D95C5C",
  "#8B0000"   # darker red extreme
))

balance_cols <- balance_palette(201)

# choose 4 colors away from the white center
pal9 <- balance_palette(9)
case_cols <- c(
  "Symmetric normal"     = pal9[2],
  "Non-symmetric normal" = pal9[3],
  "Triangular"           = pal9[7],
  "Modular"              = pal9[8]
)

# ============================================================
# Layouts
# ============================================================

circular_layout <- function(n, radius = 1.0, rotation = 0.0) {
  k <- seq_len(n)
  data.frame(
    node = k,
    x = radius * cos(rotation + 2 * pi * (k - 1) / n),
    y = radius * sin(rotation + 2 * pi * (k - 1) / n)
  )
}

balanced_oval_layout <- function(n) {
  theta <- seq(0, 2 * pi, length.out = n + 1)[-(n + 1)]
  data.frame(
    node = seq_len(n),
    x = 1.15 * cos(theta),
    y = 0.82 * sin(theta)
  )
}

triangular_layout <- function() {
  data.frame(
    node = 1:8,
    x = c(0.0, 0.2, 1.2, 1.4, 2.5, 2.7, 3.8, 4.0),
    y = c(0.55, -0.15, 0.95, 0.20, 0.35, -0.45, 0.75, -0.05)
  )
}

modular4_layout <- function() {
  data.frame(
    node = 1:8,
    x = c(-2.2, -1.75, -2.0, -1.55, 1.55, 2.0, 1.8, 2.25),
    y = c(0.95, 0.55, -0.45, -0.85, 0.85, 0.45, -0.55, -0.95)
  )
}
network_limits <- function() {
  list(
    x = c(-2.5, 4.2),
    y = c(-1.2, 1.2)
  )
}
network_layout <- function(name, n) {
  if (name == "Symmetric normal") {
    circular_layout(n, radius = 1.0, rotation = pi / 8)
  } else if (name == "Non-symmetric normal") {
    balanced_oval_layout(n)
  } else if (name == "Triangular") {
    triangular_layout()
  } else if (name == "Modular") {
    modular4_layout()
  } else {
    circular_layout(n)
  }
}

# ============================================================
# Plot builders
# ============================================================

make_edge_df <- function(A, name, tol = 1e-12) {
  G <- interaction_matrix(A)
  n <- nrow(G)
  pts <- network_layout(name, n)
  
  edges <- list()
  idx <- 1L
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i != j && abs(G[i, j]) > tol) {
        edges[[idx]] <- data.frame(
          from = j,
          to = i,
          x = pts$x[pts$node == j],
          y = pts$y[pts$node == j],
          xend = pts$x[pts$node == i],
          yend = pts$y[pts$node == i],
          value = G[i, j]
        )
        idx <- idx + 1L
      }
    }
  }
  
  edges_df <- do.call(rbind, edges)
  list(edges = edges_df, nodes = pts)
}

make_network_plot <- function(A, name, mx) {
  tmp <- make_edge_df(A, name)
  edges <- tmp$edges
  nodes <- tmp$nodes
  lims <- network_limits()
  
  ggplot() +
    geom_segment(
      data = edges,
      aes(x = x, y = y, xend = xend, yend = yend, colour = value),
      linewidth = 1.2,
      lineend = "round",
      show.legend = FALSE
    ) +
    geom_point(
      data = nodes,
      aes(x = x, y = y),
      colour = "black",
      size = 3.4
    ) +
    coord_equal(
      xlim = lims$x,
      ylim = lims$y,
      expand = FALSE,
      clip = "off"
    ) +
    scale_colour_gradient2(
      low = "#102A83",
      mid = "#F2F2F2",
      high = "#6E0000",
      midpoint = 0,
      limits = c(-mx, mx)
    ) +
    theme_void() +
    theme(
      plot.margin = margin(2, 2, 2, 2)
    )
}

make_heatmap_plot <- function(A, name, mx) {
  n <- nrow(A)
  df <- expand.grid(i = 1:n, j = 1:n)
  df$value <- as.vector(A)
  label_fill <- case_cols[[name]]
  
  ggplot(df, aes(x = j, y = i, fill = value)) +
    geom_tile() +
    scale_y_reverse(expand = expansion(mult = c(0, 0.02))) +
    scale_x_continuous(expand = c(0, 0), breaks = NULL) +
    scale_fill_gradient2(
      low = "#102A83",
      mid = "#F2F2F2",
      high = "#6E0000",
      midpoint = 0,
      limits = c(-mx, mx),
      name = "Interaction strength",
      guide = guide_colorbar(
        title.position = "right",
        title.theme = element_text(angle = 90, hjust = 0.5, vjust = 0.5),
        barheight = unit(80, "mm")
      )
    ) +
    annotate(
      "label",
      x = (n + 1) / 2,
      y = -0.55,
      label = name,
      fill = label_fill,
      colour = "white",
      fontface = "bold",
      label.size = 0.25,
      size = 4.1
    ) +
    coord_fixed(clip = "off") +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid = element_blank(),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      axis.title = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      plot.margin = margin(20, 2, 2, 2),
      legend.position = "right",
      legend.title = element_text(angle = 90, hjust = 0.5, vjust = 0.5),
      legend.text = element_text(size = 10)
    )
}

# ============================================================
# Build plot data
# ============================================================

allvals <- unlist(lapply(cases, function(z) as.vector(z$A)))
mx <- max(abs(allvals))

intr_df <- do.call(
  rbind,
  lapply(cases, function(z) {
    data.frame(
      omega = omegas,
      sensitivity = intr_profiles[[z$name]],
      architecture = z$name
    )
  })
)

intr_df$architecture <- factor(
  intr_df$architecture,
  levels = vapply(cases, function(z) z$name, character(1))
)

# ============================================================
# Assemble plots
# ============================================================

network_plots <- lapply(cases, function(z) make_network_plot(z$A, z$name, mx))
heatmap_plots <- lapply(cases, function(z) make_heatmap_plot(z$A, z$name, mx))

network_row <- wrap_plots(network_plots, nrow = 1)

heatmap_row <- wrap_plots(heatmap_plots, nrow = 1, guides = "collect") &
  theme(
    legend.position = "right",
    legend.title = element_text(angle = 90, hjust = 0.5, vjust = 0.5)
  )

p_intrinsic <- ggplot(intr_df, aes(x = omega, y = sensitivity, colour = architecture)) +
  geom_line(linewidth = 1.2) +
  scale_x_log10(
    labels = function(x) {
      out <- format(round(x, 3), scientific = FALSE, trim = TRUE)
      sub("\\.?0+$", "", out)
    }
  ) +
  scale_colour_manual(values = case_cols) +
  labs(
    # title = "Intrinsic sensitivity",
    x = expression(omega),
    y = expression("||S(" * omega * ")||"[2])
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background = element_rect(fill = "white", colour = NA),
    plot.title = element_text(face = "bold"),
    legend.title = element_blank(),
    legend.position = "right"
  )

final_plot <- network_row / heatmap_row / p_intrinsic +
  plot_layout(heights = c(0.62, 1.15, 0.85))

print(final_plot)

# Optional save:
ggsave(
  "RPlots/four_architectures_same_resilience_R.png",
  final_plot,
  width = 16,
  height = 10,
  dpi = 300
)