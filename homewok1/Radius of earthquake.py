t_sp_1 = 5 # uint: s
t_sp_2 = 7.5 # uint: s
t_sp_3 = 6 # uint: s
v_p = 8.5 # uint: km/s
v_s = 4.3 # uint: km/s

def theory_r(t_sp, v_s, v_p, n):
    r = t_sp /(1/ v_s - 1/ v_p)
    print(f'The radius of R{n} from theory solution is {r:.4f} km.')
    return r

def rule_of_thumb(t_sp, n):
    r = t_sp * 7.42
    print(f'The radius of R{n} from rule of thumb is {r:4f} km.')
    return r
def calculate_error(r1, r2):
    error = abs(r1 - r2) / r1 *100
    print(f'Relative error (based on theory solution) = {error:.3f}%.')
    return error

r1 = theory_r(t_sp_1, v_s, v_p, 1)
r2 = theory_r(t_sp_2, v_s, v_p, 2)
r3 = theory_r(t_sp_3, v_s, v_p, 3)
r1_thumb = rule_of_thumb(t_sp_1, 1)
r2_thumb = rule_of_thumb(t_sp_2, 2)
r3_thumb = rule_of_thumb(t_sp_3, 3)
error1 = calculate_error(r1, r1_thumb)
error2 = calculate_error(r2, r2_thumb)
error3 = calculate_error(r3, r3_thumb)

