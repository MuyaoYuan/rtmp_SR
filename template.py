def set_template(args):
    if args.template == 'ESPCN':
        args.model = 'ESPCN'
        args.n_colors = 3
        args.scale = 2

    elif args.template == 'ESPCN_modified':
        args.model = 'ESPCN_modified'
        args.n_colors = 3
        args.scale = 2
    
    elif args.template == 'ESPCN_multiframe':
        args.model = 'ESPCN_multiframe'
        args.n_colors = 3
        args.scale = 2
        args.n_sequence = 3
        
    else:
        print('Please Enter Appropriate Template!!!')
