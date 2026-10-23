program gyro
  implicit none
  real(8) :: m_p, q, v, B, r_g

  m_p = 1.6726d-27
  q   = 1.6022d-19 
  v   = 400.0d3 !SW speed (m/s)
  B   = 54.4d-9 !standoff field (T)

  r_g = m_p * v / (q * B)
  print *, 'Proton gyroradius (km):', r_g / 1.0d3
end program gyro